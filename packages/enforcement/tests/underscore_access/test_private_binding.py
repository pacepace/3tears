"""the private-binding gate, against synthetic repos whose every binding is known.

Each test builds the smallest repo that exhibits one spelling, so a rule that stopped seeing it
fails by name, and each allowed spelling has its own test so a rule that widened to flag it fails
by name too. The scan-count tests are the non-vacuity guard on the walker's inputs: an empty scan
and a recogniser that matched no call both report what a compliant repo reports.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.underscore_access import (
    PRIVATE_BINDING_CATEGORIES,
    SHAPE_G_MODULE,
    SHAPE_G_NAME,
    SHAPE_H_ATTRIBUTE,
    SHAPE_H_PATH,
    private_binding_findings,
    scan_private_bindings,
    undetected_planted_controls,
)

_INTERNALS = "thing = 1\n_secret = 2\n"
_MOD = "public = 1\n_hidden = 2\n\n\nclass Client:\n    def _astream(self) -> None:\n        pass\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _repo(tmp_path: Path, *, ignores: dict[str, str] | None = None, ledger: str = "") -> tuple[Path, Path]:
    """a single-package repo: ``src/pkg`` with a private module, a ruff config and a ledger.

    :param tmp_path: pytest's temporary directory
    :ptype tmp_path: Path
    :param ignores: per-file-ignores keys to the code list they carry
    :ptype ignores: dict[str, str] | None
    :param ledger: the exemptions ledger's text
    :ptype ledger: str
    :return: the repo root and the ledger path
    :rtype: tuple[Path, Path]
    """
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    lines = ["[tool.ruff.lint.per-file-ignores]"]
    lines += [f'"{key}" = {codes}' for key, codes in (ignores or {}).items()]
    _write(repo / "pyproject.toml", "\n".join(lines) + "\n")
    _write(repo / "src" / "pkg" / "__init__.py", "")
    _write(repo / "src" / "pkg" / "_internals.py", _INTERNALS)
    _write(repo / "src" / "pkg" / "mod.py", _MOD)
    _write(repo / "src" / "other" / "__init__.py", "")
    exemptions = _write(repo / "tests" / "enforcement" / "_underscore_exemptions.txt", ledger)
    return repo, exemptions


def _found(repo: Path, exemptions: Path | None = None) -> set[tuple[str, str, int, str]]:
    """``(category, relative path, line, symbol)`` for every violation in *repo*."""
    scan = scan_private_bindings(repo, exemptions)
    return {(v.category, v.file.relative_to(repo).as_posix(), v.line, v.symbol) for v in scan.violations}


class TestPrivateImports:
    def test_a_test_importing_a_private_name_from_src_is_a_violation(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "test_a.py", "from pkg.mod import _hidden\n")

        assert _found(repo) == {(SHAPE_G_NAME, "tests/test_a.py", 1, "_hidden")}

    def test_a_test_importing_a_private_helper_from_another_test_module_is_a_violation(self, tmp_path: Path) -> None:
        """the test module is the helper's owner; a sibling test is outside it."""
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "__init__.py", "")
        _write(repo / "tests" / "test_websocket.py", "def _valid_auth() -> None:\n    pass\n")
        _write(repo / "tests" / "test_b.py", "from .test_websocket import _valid_auth\n")

        assert _found(repo) == {(SHAPE_G_NAME, "tests/test_b.py", 1, "_valid_auth")}

    def test_a_private_support_module_in_the_same_tests_tree_is_allowed(self, tmp_path: Path) -> None:
        """an underscore on a test-support module marks it non-collected support, not another owner's API."""
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "__init__.py", "")
        _write(repo / "tests" / "support" / "__init__.py", "")
        _write(repo / "tests" / "support" / "_pod_auth.py", "def make_auth() -> None:\n    pass\n")
        _write(
            repo / "tests" / "unit" / "test_c.py",
            "from tests.support._pod_auth import make_auth\nimport tests.support._pod_auth\n",
        )
        _write(repo / "tests" / "support" / "test_d.py", "from ._pod_auth import make_auth\nfrom . import _pod_auth\n")

        assert _found(repo) == set()

    def test_a_private_name_from_a_private_support_module_is_still_a_violation(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "support" / "_pod_auth.py", "def _sign() -> None:\n    pass\n")
        _write(repo / "tests" / "support" / "test_e.py", "from ._pod_auth import _sign\n")

        assert _found(repo) == {(SHAPE_G_NAME, "tests/support/test_e.py", 1, "_sign")}

    def test_a_test_importing_through_a_private_src_module_is_a_violation(self, tmp_path: Path) -> None:
        """the public name does not make the private module's path a stable thing to bind to."""
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "test_f.py", "from pkg._internals import thing\nimport pkg._internals as internals\n")

        assert _found(repo) == {
            (SHAPE_G_MODULE, "tests/test_f.py", 1, "_internals"),
            (SHAPE_G_MODULE, "tests/test_f.py", 2, "_internals"),
        }

    def test_a_private_module_imported_by_name_from_its_package_is_a_module_violation(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "test_g.py", "from pkg import _internals\n")

        assert _found(repo) == {(SHAPE_G_MODULE, "tests/test_g.py", 1, "_internals")}

    def test_a_third_party_private_module_is_a_violation_in_tests(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "test_h.py", "from _pytest.monkeypatch import MonkeyPatch\n")

        assert _found(repo) == {(SHAPE_G_MODULE, "tests/test_h.py", 1, "_pytest")}

    def test_src_importing_a_private_name_inside_its_own_package_is_allowed(self, tmp_path: Path) -> None:
        """the existing contract: a package's privates are shared by that package's modules."""
        repo, _ = _repo(tmp_path)
        _write(repo / "src" / "pkg" / "user.py", "from pkg._internals import _secret\nfrom ._internals import thing\n")

        assert _found(repo) == set()

    def test_src_importing_a_private_name_from_another_package_is_a_violation(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "src" / "other" / "user.py", "from pkg.mod import _hidden\n")

        assert _found(repo) == {(SHAPE_G_NAME, "src/other/user.py", 1, "_hidden")}

    def test_src_importing_a_third_party_private_is_a_violation(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "src" / "pkg" / "client.py", "from vendor.client import _transport\n")

        assert _found(repo) == {(SHAPE_G_NAME, "src/pkg/client.py", 1, "_transport")}

    def test_a_recorded_confinement_module_may_import_third_party_privates(self, tmp_path: Path) -> None:
        """the only sanctioned private access: one src module per library, ignored and recorded."""
        module = "src/pkg/_vendor_internals.py"
        ledger = (
            f"# rationale: vendor keeps the socket on Client._sock with no public accessor\n{module}:sock#0:_sock\n"
        )
        repo, exemptions = _repo(tmp_path, ignores={module: '["SLF001"]'}, ledger=ledger)
        _write(repo / "src" / "other" / "_x.py", "y = 1\n")
        _write(
            repo / module,
            "from vendor._internal.query import Query\nfrom vendor import _private_helper\nfrom other._x import y\n",
        )

        assert _found(repo, exemptions) == {(SHAPE_G_MODULE, module, 3, "_x")}

    def test_a_confinement_module_without_a_ledger_entry_is_not_sanctioned(self, tmp_path: Path) -> None:
        module = "src/pkg/_vendor_internals.py"
        repo, exemptions = _repo(tmp_path, ignores={module: '["SLF001"]'})
        _write(repo / module, "from vendor._internal.query import Query\n")

        assert _found(repo, exemptions) == {(SHAPE_G_MODULE, module, 1, "_internal")}

    def test_dunders_are_not_private(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "test_i.py", "from __future__ import annotations\nfrom pkg import __version__\n")

        assert _found(repo) == set()


class TestStringBindings:
    def test_monkeypatch_setattr_on_an_object_by_private_name(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(
            repo / "tests" / "test_a.py", 'def test_x(monkeypatch, obj):\n    monkeypatch.setattr(obj, "_state", 1)\n'
        )

        assert _found(repo) == {(SHAPE_H_ATTRIBUTE, "tests/test_a.py", 2, "_state")}

    def test_monkeypatch_setattr_and_delattr_by_dotted_path(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(
            repo / "tests" / "test_b.py",
            "def test_x(monkeypatch):\n"
            '    monkeypatch.setattr("pkg.mod._hidden", 3)\n'
            '    monkeypatch.delattr("pkg._internals.thing")\n'
            '    monkeypatch.delattr(object(), "_gone")\n',
        )

        assert _found(repo) == {
            (SHAPE_H_PATH, "tests/test_b.py", 2, "_hidden"),
            (SHAPE_H_PATH, "tests/test_b.py", 3, "_internals"),
            (SHAPE_H_ATTRIBUTE, "tests/test_b.py", 4, "_gone"),
        }

    def test_every_patch_spelling(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(
            repo / "tests" / "test_c.py",
            "from unittest import mock\n"
            "from unittest.mock import patch\n"
            "from unittest.mock import patch as mpatch\n"
            "from pkg.mod import Client\n"
            '@patch("pkg.mod._hidden")\n'
            "def test_x(mocker):\n"
            '    mock.patch("pkg._internals.thing")\n'
            '    mpatch("pkg.mod._hidden")\n'
            '    patch.object(Client, "_astream")\n'
            '    mocker.patch.object(Client, "_astream")\n'
            '    mocker.patch("pkg.mod._hidden")\n'
            '    patch.multiple("pkg.mod", _hidden=1)\n'
            '    patch.dict("pkg.mod._REGISTRY", {})\n'
            '    mocker.spy(Client, "_astream")\n'
            '    mock.patch.object(target=Client, attribute="_astream")\n',
        )

        assert _found(repo) == {
            (SHAPE_H_PATH, "tests/test_c.py", 5, "_hidden"),
            (SHAPE_H_PATH, "tests/test_c.py", 7, "_internals"),
            (SHAPE_H_PATH, "tests/test_c.py", 8, "_hidden"),
            (SHAPE_H_ATTRIBUTE, "tests/test_c.py", 9, "_astream"),
            (SHAPE_H_ATTRIBUTE, "tests/test_c.py", 10, "_astream"),
            (SHAPE_H_PATH, "tests/test_c.py", 11, "_hidden"),
            (SHAPE_H_PATH, "tests/test_c.py", 12, "_hidden"),
            (SHAPE_H_PATH, "tests/test_c.py", 13, "_REGISTRY"),
            (SHAPE_H_ATTRIBUTE, "tests/test_c.py", 14, "_astream"),
            (SHAPE_H_ATTRIBUTE, "tests/test_c.py", 15, "_astream"),
        }

    def test_patching_a_third_party_private_is_a_violation(self, tmp_path: Path) -> None:
        """a test's patch is never a confinement module, whatever library it reaches into."""
        repo, _ = _repo(tmp_path)
        _write(
            repo / "tests" / "test_d.py",
            "from unittest.mock import patch\n"
            "from langchain_anthropic import ChatAnthropic\n"
            'patch.object(ChatAnthropic, "_astream")\n'
            'patch("langchain_anthropic.chat_models.ChatAnthropic._astream")\n',
        )

        assert _found(repo) == {
            (SHAPE_H_ATTRIBUTE, "tests/test_d.py", 3, "_astream"),
            (SHAPE_H_PATH, "tests/test_d.py", 4, "_astream"),
        }

    def test_import_module_by_private_path(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "test_e.py", 'import importlib\nimportlib.import_module("pkg._internals")\n')

        assert _found(repo) == {(SHAPE_H_PATH, "tests/test_e.py", 2, "_internals")}

    def test_the_allowed_spellings(self, tmp_path: Path) -> None:
        """the owner, the module's own definitions, public and dunder names, and non-patch calls."""
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "__init__.py", "")
        _write(repo / "tests" / "support" / "__init__.py", "")
        _write(repo / "tests" / "support" / "_clock.py", "def now() -> int:\n    return 0\n")
        _write(
            repo / "tests" / "test_f.py",
            "from unittest.mock import patch\n"
            "class FakeThing:\n"
            "    _state = 0\n"
            "class TestX:\n"
            "    def test_x(self, monkeypatch, client, obj):\n"
            '        monkeypatch.setattr(self, "_seen", 1)\n'
            '        monkeypatch.setattr(FakeThing, "_state", 1)\n'
            '        patch("pkg.mod.public")\n'
            '        patch("tests.support._clock.now")\n'
            '        patch("tests.test_f._own_helper")\n'
            '        client.patch("/api/v1/_x")\n'
            '        monkeypatch.setattr(obj, "__init__", None)\n'
            '        monkeypatch.setattr("os.environ", {})\n'
            "def _own_helper() -> None:\n"
            "    pass\n",
        )

        assert _found(repo) == set()

    def test_a_class_binding_its_own_private_on_a_fresh_instance_is_the_owner(self, tmp_path: Path) -> None:
        """the frozen-slots constructor: the enclosing class declares the private it sets."""
        repo, _ = _repo(tmp_path)
        _write(
            repo / "src" / "pkg" / "scope.py",
            "class Scope:\n"
            '    __slots__ = ("_reason",)\n'
            "    _customer_id: int\n"
            "    def __init__(self) -> None:\n"
            "        self._config_key = None\n"
            "    @classmethod\n"
            "    def create(cls) -> 'Scope':\n"
            "        scope = object.__new__(cls)\n"
            '        object.__setattr__(scope, "_reason", 1)\n'
            '        object.__setattr__(scope, "_customer_id", 1)\n'
            '        object.__setattr__(scope, "_config_key", 1)\n'
            '        object.__setattr__(scope, "_undeclared", 1)\n'
            "        return scope\n"
            "def outside(scope: Scope) -> None:\n"
            '    object.__setattr__(scope, "_reason", 2)\n',
        )

        assert _found(repo) == {
            (SHAPE_H_ATTRIBUTE, "src/pkg/scope.py", 12, "_undeclared"),
            (SHAPE_H_ATTRIBUTE, "src/pkg/scope.py", 15, "_reason"),
        }

    def test_a_patch_name_not_imported_from_mock_is_not_a_patch(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / "tests" / "test_g.py", 'def patch(x):\n    return x\npatch("pkg.mod._hidden")\n')

        assert _found(repo) == set()


class TestTheScanReportsItsInputs:
    def test_files_imports_and_binding_calls_are_counted(self, tmp_path: Path) -> None:
        """the non-vacuity inputs a consumer's floor asserts on."""
        repo, exemptions = _repo(tmp_path)
        _write(
            repo / "tests" / "test_a.py",
            'import os\nfrom pkg import mod\ndef test_x(monkeypatch):\n    monkeypatch.setattr("os.sep", "/")\n',
        )

        scan = scan_private_bindings(repo, exemptions)

        assert scan.files_scanned == 5
        assert scan.imports_examined == 2
        assert scan.binding_calls_examined == 1
        assert scan.violations == ()

    def test_a_vendored_tree_is_not_scanned(self, tmp_path: Path) -> None:
        repo, _ = _repo(tmp_path)
        _write(repo / ".venv" / "lib" / "dep.py", "from pkg.mod import _hidden\n")

        assert _found(repo) == set()

    def test_findings_name_the_place_and_the_fix(self, tmp_path: Path) -> None:
        repo, exemptions = _repo(tmp_path)
        _write(repo / "tests" / "test_a.py", "from pkg.mod import _hidden\n")

        findings = private_binding_findings(scan_private_bindings(repo, exemptions), repo)

        assert len(findings) == 1
        assert findings[0].startswith(f"[{SHAPE_G_NAME}] tests/test_a.py:1:_hidden")
        assert "front door" in findings[0]


class TestPlantedControls:
    def test_every_shape_is_detected(self, tmp_path: Path) -> None:
        assert undetected_planted_controls(tmp_path) == []

    def test_the_categories_are_the_four_sub_shapes(self) -> None:
        assert PRIVATE_BINDING_CATEGORIES == (SHAPE_G_NAME, SHAPE_G_MODULE, SHAPE_H_ATTRIBUTE, SHAPE_H_PATH)
