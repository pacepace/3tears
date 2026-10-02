"""the SLF001 suppression policy, against synthetic repos whose every violation is known.

Each test builds the smallest repo that exhibits one spelling, so a check that stopped seeing it
fails by name. The clean-repo test asserts both inputs are populated before it trusts an empty
verdict: an empty scan and an empty config read report exactly what a compliant repo reports.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.underscore_access import (
    is_src_module,
    ledger_entries_outside_src,
    scanned_python_files,
    slf001_ignored_files,
    slf001_ignores_outside_src,
    slf001_ignores_without_a_ledger_entry,
    slf001_policy_findings,
    slf001_pragma_offenders,
)

_SRC_MODULE = "packages/lib/src/lib/_vendor_internals.py"
_PRIVATE_READ = "value = client._pending\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _repo(tmp_path: Path, *, ignores: dict[str, str] | None = None, ledger: str = "") -> tuple[Path, Path]:
    """a repo root with a root ruff config and an exemptions ledger.

    :param tmp_path: pytest's temporary directory
    :ptype tmp_path: Path
    :param ignores: per-file-ignores keys to the code list they carry
    :ptype ignores: dict[str, str] | None
    :param ledger: the ledger's text
    :ptype ledger: str
    :return: the repo root and the ledger path
    :rtype: tuple[Path, Path]
    """
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    lines = ["[tool.ruff.lint.per-file-ignores]"]
    lines += [f'"{key}" = {codes}' for key, codes in (ignores or {}).items()]
    _write(repo / "pyproject.toml", "\n".join(lines) + "\n")
    exemptions = _write(repo / "tests" / "enforcement" / "_underscore_exemptions.txt", ledger)
    return repo, exemptions


def _offending_lines(repo: Path) -> set[str]:
    return {offender.split(" -- ")[0] for offender in slf001_pragma_offenders(repo)}


class TestInlinePragmas:
    def test_a_code_list_naming_slf001_is_an_offender(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "test_a.py", "value = client._pending  # noqa: SLF001 -- reason\n")

        assert _offending_lines(repo) == {"tests/test_a.py:1"}

    def test_every_code_list_spelling_is_an_offender(self, tmp_path: Path) -> None:
        """the group, a mixed list, and the ruff-prefixed form all silence SLF001."""
        repo, _ = _repo(tmp_path)
        _write(
            repo / "pkg" / "mod.py",
            "a = x._a  # noqa: SLF\nb = x._b  # noqa: E501, SLF001\nc = x._c  # ruff: noqa: SLF001\n",
        )

        assert _offending_lines(repo) == {"pkg/mod.py:1", "pkg/mod.py:2", "pkg/mod.py:3"}

    def test_a_pragma_on_a_line_with_no_private_access_is_still_an_offender(self, tmp_path: Path) -> None:
        """a code list naming SLF001 is the banned spelling wherever it sits."""
        repo, _ = _repo(tmp_path)
        _write(repo / "pkg" / "mod.py", "from pkg._private import thing  # noqa: SLF001\n")

        assert _offending_lines(repo) == {"pkg/mod.py:1"}

    def test_a_bare_noqa_on_a_private_access_is_an_offender(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "pkg" / "mod.py", "value = client._pending  # noqa\n")

        assert _offending_lines(repo) == {"pkg/mod.py:1"}

    def test_a_bare_noqa_on_a_line_without_a_private_access_is_not_this_policys(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "pkg" / "mod.py", "import os  # noqa\nvalue = self._own\n")

        assert slf001_pragma_offenders(repo) == []

    def test_a_file_level_bare_noqa_in_a_file_with_a_private_access_is_an_offender(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "pkg" / "mod.py", "# ruff: noqa\n" + _PRIVATE_READ)

        assert _offending_lines(repo) == {"pkg/mod.py:1"}

    def test_another_rules_code_list_is_not_an_offender(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "pkg" / "mod.py", "value = client._pending  # noqa: BLE001\n")

        assert slf001_pragma_offenders(repo) == []

    def test_the_spelling_quoted_in_a_string_is_not_a_pragma(self, tmp_path: Path) -> None:
        """a docstring documenting the banned spelling -- this policy's own -- suppresses nothing."""
        repo, _ = _repo(tmp_path)
        _write(repo / "pkg" / "mod.py", '"""never write ``# noqa: SLF001``."""\nTEXT = "# noqa: SLF001"\n')

        assert slf001_pragma_offenders(repo) == []

    def test_vendored_trees_and_nested_checkouts_are_not_scanned(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        pragma = "value = client._pending  # noqa: SLF001\n"
        _write(repo / ".venv" / "lib" / "dep.py", pragma)
        worktree = repo / ".claude" / "worktrees" / "wt"
        _write(worktree / ".git", "gitdir: elsewhere\n")
        _write(worktree / "pkg" / "mod.py", pragma)
        _write(repo / "pkg" / "own.py", pragma)

        assert _offending_lines(repo) == {"pkg/own.py:1"}
        scanned = {path.relative_to(repo).as_posix() for path in scanned_python_files(repo)}
        assert scanned == {"pkg/own.py"}


class TestPerFileIgnores:
    def test_an_ignore_on_a_test_file_is_a_finding(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path, ignores={"tests/test_a.py": '["SLF001"]'})
        _write(repo / "tests" / "test_a.py", _PRIVATE_READ)

        assert slf001_ignores_outside_src(repo) == [
            "pyproject.toml: 'tests/test_a.py' covers tests/test_a.py, which is not a src module"
        ]

    def test_a_glob_covering_tests_is_a_finding_for_each_file(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path, ignores={"**/tests/**": '["N803", "SLF001"]'})
        _write(repo / "packages" / "lib" / "tests" / "test_a.py", _PRIVATE_READ)
        _write(repo / "packages" / "lib" / "tests" / "unit" / "test_b.py", _PRIVATE_READ)

        assert len(slf001_ignores_outside_src(repo)) == 2

    def test_an_ignore_for_another_rule_is_none_of_this_policys_business(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path, ignores={"**/tests/**": '["N803", "N806"]'})
        _write(repo / "tests" / "test_a.py", _PRIVATE_READ)

        assert slf001_ignored_files(repo) == []
        assert slf001_ignores_outside_src(repo) == []

    def test_a_nested_ruff_config_is_read_too(self, tmp_path: Path) -> None:
        """a nested ruff.toml is a full override; reading only the root would miss its ignores."""
        repo, _ = _repo(tmp_path)
        _write(repo / "sidecar" / "ruff.toml", '[lint.per-file-ignores]\n"tests/*" = ["SLF001"]\n')
        _write(repo / "sidecar" / "tests" / "test_x.py", _PRIVATE_READ)

        assert slf001_ignores_outside_src(repo) == [
            "sidecar/ruff.toml: 'tests/*' covers sidecar/tests/test_x.py, which is not a src module"
        ]

    def test_a_recorded_src_module_is_clean(self, tmp_path: Path) -> None:
        ledger = f"# rationale: the vendor library has no public pending buffer\n{_SRC_MODULE}:<module>#0:_pending\n"
        repo, exemptions = _repo(tmp_path, ignores={_SRC_MODULE: '["SLF001"]'}, ledger=ledger)
        _write(repo / _SRC_MODULE, _PRIVATE_READ)

        assert slf001_ignores_outside_src(repo) == []
        assert slf001_ignores_without_a_ledger_entry(repo, exemptions) == []

    def test_an_unrecorded_src_module_is_a_finding(self, tmp_path: Path) -> None:
        repo, exemptions = _repo(tmp_path, ignores={_SRC_MODULE: '["SLF001"]'})
        _write(repo / _SRC_MODULE, _PRIVATE_READ)

        assert slf001_ignores_without_a_ledger_entry(repo, exemptions) == [
            f"pyproject.toml: {_SRC_MODULE!r} covers {_SRC_MODULE}, which has no entry in _underscore_exemptions.txt"
        ]


class TestLedgerEntries:
    def test_an_entry_for_a_test_file_is_a_finding(self, tmp_path: Path) -> None:
        ledger = (
            "# rationale: tests need access\n"
            "packages/lib/tests/test_a.py:TestA.test_it#0:_pending\n"
            "# rationale: the vendor library has no public pending buffer\n"
            f"{_SRC_MODULE}:<module>#0:_pending\n"
        )
        repo, exemptions = _repo(tmp_path, ledger=ledger)

        assert ledger_entries_outside_src(exemptions, repo) == ["packages/lib/tests/test_a.py"]

    def test_a_confinement_modules_own_test_importing_it_is_not_a_finding(self, tmp_path: Path) -> None:
        """owner ruling 1: the module's own test exists to catch the library changing, so it may import it."""
        own_test = "packages/lib/tests/unit/test_vendor_internals.py"
        ledger = (
            "# rationale: the vendor library has no public pending buffer\n"
            f"{_SRC_MODULE}:<module>#0:_pending\n"
            "# rationale: pins that the vendor client still carries _pending, which the confinement module reads\n"
            f"{own_test}:<module>#0:_vendor_internals\n"
        )
        repo, exemptions = _repo(tmp_path, ignores={_SRC_MODULE: '["SLF001"]'}, ledger=ledger)
        _write(repo / _SRC_MODULE, _PRIVATE_READ)

        assert ledger_entries_outside_src(exemptions, repo) == []

    def test_any_other_test_entry_beside_a_confinement_module_is_still_a_finding(self, tmp_path: Path) -> None:
        """only the own test, and only the import of the module itself: not a sibling test, not another name."""
        own_test = "packages/lib/tests/unit/test_vendor_internals.py"
        sibling = "packages/lib/tests/unit/test_driver.py"
        ledger = (
            "# rationale: the vendor library has no public pending buffer\n"
            f"{_SRC_MODULE}:<module>#0:_pending\n"
            "# rationale: the driver test wants the confinement module too\n"
            f"{sibling}:<module>#0:_vendor_internals\n"
            "# rationale: the own test reads a private member directly\n"
            f"{own_test}:test_it#0:_pending\n"
        )
        repo, exemptions = _repo(tmp_path, ignores={_SRC_MODULE: '["SLF001"]'}, ledger=ledger)
        _write(repo / _SRC_MODULE, _PRIVATE_READ)

        assert ledger_entries_outside_src(exemptions, repo) == [sibling, own_test]

    def test_an_own_test_entry_for_a_module_that_is_not_a_confinement_module_is_a_finding(self, tmp_path: Path) -> None:
        """no per-file SLF001 ignore means no recorded confinement module, so nothing sanctions its test."""
        own_test = "packages/lib/tests/unit/test_vendor_internals.py"
        ledger = f"# rationale: pins the vendor members\n{own_test}:<module>#0:_vendor_internals\n"
        repo, exemptions = _repo(tmp_path, ledger=ledger)
        _write(repo / _SRC_MODULE, _PRIVATE_READ)

        assert ledger_entries_outside_src(exemptions, repo) == [own_test]


class TestSrcModuleClassification:
    def test_the_shapes_that_are_and_are_not_src(self) -> None:
        assert is_src_module("packages/nats/src/threetears/nats/_nats_py_internals.py")
        assert is_src_module("src/aibots/hub/app.py")
        assert not is_src_module("packages/nats/tests/unit/test_client.py")
        assert not is_src_module("packages/nats/src/threetears/nats/tests/helper.py")
        assert not is_src_module("src/pkg/test_something.py")
        assert not is_src_module("src/pkg/conftest.py")
        assert not is_src_module("scripts/regen.py")


class TestTheCombinedVerdict:
    def test_a_compliant_repo_has_no_findings_with_both_inputs_populated(self, tmp_path: Path) -> None:
        """non-vacuity on both inputs: files were scanned AND an ignore was read, then nothing found."""
        ledger = f"# rationale: the vendor library has no public pending buffer\n{_SRC_MODULE}:<module>#0:_pending\n"
        repo, exemptions = _repo(tmp_path, ignores={_SRC_MODULE: '["SLF001"]'}, ledger=ledger)
        _write(repo / _SRC_MODULE, _PRIVATE_READ)
        _write(repo / "packages" / "lib" / "tests" / "test_a.py", "def test_it() -> None:\n    assert True\n")

        assert len(scanned_python_files(repo)) == 2
        assert [path for _config, _key, path in slf001_ignored_files(repo)] == [_SRC_MODULE]
        assert slf001_policy_findings(repo, exemptions) == []

    def test_every_kind_of_breach_reaches_the_verdict(self, tmp_path: Path) -> None:
        ledger = "# rationale: tests need access\npackages/lib/tests/test_a.py:test_it#0:_pending\n"
        repo, exemptions = _repo(
            tmp_path,
            ignores={_SRC_MODULE: '["SLF001"]', "packages/lib/tests/test_a.py": '["SLF001"]'},
            ledger=ledger,
        )
        _write(repo / _SRC_MODULE, _PRIVATE_READ)
        _write(repo / "packages" / "lib" / "tests" / "test_a.py", "def test_it() -> None:\n    client._pending\n")
        _write(repo / "packages" / "lib" / "src" / "lib" / "other.py", "value = c._x  # noqa: SLF001\n")

        findings = slf001_policy_findings(repo, exemptions)

        kinds = sorted(finding.split(":", 1)[0] for finding in findings)
        assert kinds == [
            "exemptions-ledger entry for a test file",
            "inline SLF001 suppression",
            "per-file SLF001 ignore on a test file",
            "unrecorded per-file SLF001 ignore",
        ]
