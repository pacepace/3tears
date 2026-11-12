"""Enforcement: a test that reads a credential from the environment carries the ``secrets`` marker.

The testament pool ships no credentials to its runners, and ``.testament.toml``'s suites deselect ``secrets``.
A test that needs the Redshift warehouse password, the Claude CLI's subscription token or a Tavily API key can only
run where that credential is: on the laptop, through ``scripts/test-integration.sh`` or
``scripts/test-live-claude-cli.sh``. Without the marker the pool would collect it and skip it for the missing key, and
a skip reads like a pass.

This walker pins the contract: in any ``test_*.py`` under a package's ``tests`` tree or the repo's ``tests/``, a test
that reads one of ``_CREDENTIALS`` from the environment (``os.environ[...]``, ``os.environ.get(...)``,
``os.getenv(...)``, ``... in os.environ``, by literal or by a module constant holding the name), or requests a fixture
in ``_FIXTURES``, carries ``@pytest.mark.secrets``: on itself, on its class, or in the module's ``pytestmark``.

- A read at module level (``_TOKEN = os.environ.get(...)``) needs the module's ``pytestmark``.
- A read in a class's fixture or helper needs the class marked (decorator or class ``pytestmark``), since every test
  of the class may reach it.
- A module-level helper or fixture that reads one passes the need on to every test that names it, followed to a fixed
  point, as ``test_browser_marker.py`` follows its helpers.
- A read inside an ``assert`` is not a use of the credential (``assert os.environ.get(NAME) != token`` checks that
  nothing leaked into the environment), and ``monkeypatch.setenv`` writes rather than reads.

**Heuristic boundary.** Each file is parsed alone, so a credential read by a helper in another module, or by a conftest
fixture not named in ``_FIXTURES``, escapes the walk. ``redshift_config`` (``packages/datasources/tests/integration/
conftest.py``) resolves ``OTS_REDSHIFT_PASSWORD`` that way; add any new one beside it.

AST-based and side-effect-free.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from threetears.enforcement.common import parse_python_path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: environment variables that hold a credential, or switch on a run that spends one
_CREDENTIALS: frozenset[str] = frozenset(
    {"OTS_REDSHIFT_PASSWORD", "CLAUDE_CODE_OAUTH_TOKEN", "TAVILY_API_KEY", "THREETEARS_LIVE_CLAUDE_CLI"}
)

#: conftest fixtures that resolve a credential for the tests requesting them
_FIXTURES: frozenset[str] = frozenset({"redshift_config"})

_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _test_files() -> list[Path]:
    """Every ``test_*.py`` under a workspace package's ``tests`` tree or the repo's ``tests/``."""
    found = [*(_REPO_ROOT / "packages").glob("**/tests/**/test_*.py"), *(_REPO_ROOT / "tests").glob("**/test_*.py")]
    return sorted(path for path in found if "__pycache__" not in path.parts)


def _credential_constants(tree: ast.Module) -> set[str]:
    """Module-level names bound to a credential's variable name (``KEY_VAR = "TAVILY_API_KEY"``)."""
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value in _CREDENTIALS:
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return bound


def _is_environ(node: ast.expr) -> bool:
    """Whether ``node`` is ``os.environ`` or a bare ``environ``."""
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


class _Reads:
    """Finds the reads of a credential in a module, leaving out those inside an ``assert``."""

    def __init__(self, tree: ast.Module) -> None:
        self._constants = _credential_constants(tree)
        self._asserted = {id(sub) for node in ast.walk(tree) if isinstance(node, ast.Assert) for sub in ast.walk(node)}

    def _names_credential(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return node.value in _CREDENTIALS
        return isinstance(node, ast.Name) and node.id in self._constants

    def _is_read(self, node: ast.AST) -> bool:
        if id(node) in self._asserted:
            return False
        if isinstance(node, ast.Call) and node.args and self._names_credential(node.args[0]):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "get" and _is_environ(func.value):
                return True
            return (isinstance(func, ast.Attribute) and func.attr == "getenv") or (
                isinstance(func, ast.Name) and func.id == "getenv"
            )
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and _is_environ(node.value):
            return self._names_credential(node.slice)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            return _is_environ(node.comparators[0]) and self._names_credential(node.left)
        return False

    def within(self, node: ast.AST) -> bool:
        """Whether ``node`` (with its decorators, for a function or class) holds a read."""
        return any(self._is_read(sub) for sub in ast.walk(node))


def _secrets_mark(node: ast.expr) -> bool:
    """Whether a decorator or a ``pytestmark`` value names ``mark.secrets``."""
    return any(
        isinstance(sub, ast.Attribute)
        and sub.attr == "secrets"
        and isinstance(sub.value, ast.Attribute)
        and sub.value.attr == "mark"
        for sub in ast.walk(node)
    )


def _pytestmark_names_secrets(body: list[ast.stmt]) -> bool:
    """Whether a module's or a class's body assigns a ``pytestmark`` that names ``mark.secrets``."""
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
        and _secrets_mark(node.value)
        for node in body
    )


def _loaded_names(node: ast.AST) -> set[str]:
    """Every bare name a function loads, and every parameter it takes (a fixture is requested by its name)."""
    names = {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)}
    if isinstance(node, _Function):
        args = node.args
        names.update(p.arg for p in (*args.posonlyargs, *args.args, *args.kwonlyargs))
    return names


def _unmarked_secret_tests(tree: ast.Module) -> list[str]:
    """What in ``tree`` reads a credential without the ``secrets`` marker, by name."""
    reads = _Reads(tree)
    if _pytestmark_names_secrets(tree.body):
        return []
    offenders = [
        f"module level, line {node.lineno}"
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and reads.within(node)
    ]
    # the module's own functions that reach a credential, directly or through each other, to a fixed point
    functions = [node for node in tree.body if isinstance(node, _Function)]
    # a conftest fixture counts only where the module does not define its own of that name, which shadows it
    reaching = set(_FIXTURES - {function.name for function in functions})
    grew = True
    while grew:
        grew = False
        for function in functions:
            if function.name not in reaching and (reads.within(function) or _loaded_names(function) & reaching):
                reaching.add(function.name)
                grew = True

    def reaches(function: _Function) -> bool:
        return reads.within(function) or bool(_loaded_names(function) & reaching)

    def visit(body: list[ast.stmt], prefix: str, marked: bool) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                held = marked or _pytestmark_names_secrets(node.body) or any(map(_secrets_mark, node.decorator_list))
                members = [m for m in node.body if isinstance(m, _Function) and not m.name.startswith("test")]
                if not held and (any(map(reads.within, node.decorator_list)) or any(map(reaches, members))):
                    offenders.append(f"{prefix}{node.name} (a fixture or helper of the class reads one)")
                    held = True
                visit(node.body, f"{prefix}{node.name}::", held)
            elif isinstance(node, _Function) and node.name.startswith("test"):
                if reaches(node) and not (marked or any(map(_secrets_mark, node.decorator_list))):
                    offenders.append(f"{prefix}{node.name}")

    visit(tree.body, "", False)
    return offenders


def test_tests_that_read_a_credential_carry_the_secrets_marker() -> None:
    """A test reading a credential from the environment must be marked ``secrets``."""
    offenders: list[str] = []
    for path in _test_files():
        for name in _unmarked_secret_tests(parse_python_path(path)):
            offenders.append(f"{path.relative_to(_REPO_ROOT)}::{name}")
    if offenders:
        listing = "\n".join(f"  - {o}" for o in offenders)
        pytest.fail(
            "test(s) read a credential from the environment without `@pytest.mark.secrets`. The testament pool "
            "ships no credentials and deselects `secrets`; without the marker it runs them, they skip for the "
            "missing key, and the skip reads like a pass:\n"
            f"{listing}"
        )


_PLANTED = """
import os
import pytest

KEY_VAR = "TAVILY_API_KEY"

def _key():
    return os.environ[KEY_VAR]

@pytest.fixture
def key():
    return _key()

def test_through_a_fixture(key):
    assert key

@pytest.mark.skipif(not os.getenv("OTS_REDSHIFT_PASSWORD"), reason="no password")
def test_gated_by_a_skipif():
    pass

def test_through_a_conftest_fixture(redshift_config):
    assert redshift_config

@pytest.mark.secrets
def test_marked():
    assert "CLAUDE_CODE_OAUTH_TOKEN" in os.environ

def test_asserts_nothing_leaked(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    assert os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") != "token"

class TestLoose:
    @pytest.fixture(autouse=True)
    def _wire(self, redshift_config):
        self.config = redshift_config

class TestHeld:
    pytestmark = pytest.mark.secrets

    def test_marked_by_its_class(self, key):
        assert key
"""


def test_the_walker_finds_each_unmarked_reader_and_passes_the_rest() -> None:
    """The control: the detector flags exactly the planted unmarked readers."""
    assert _unmarked_secret_tests(ast.parse(_PLANTED)) == [
        "test_through_a_fixture",
        "test_gated_by_a_skipif",
        "test_through_a_conftest_fixture",
        "TestLoose (a fixture or helper of the class reads one)",
    ]


def test_a_module_s_own_fixture_shadows_the_conftest_one() -> None:
    """A unit test's local ``redshift_config`` builds a config for a fake; it reads no credential."""
    planted = "import pytest\n\n@pytest.fixture\ndef redshift_config():\n    return object()\n\n"
    planted += "def test_uses_the_local_one(redshift_config):\n    assert redshift_config\n"
    assert _unmarked_secret_tests(ast.parse(planted)) == []


def test_a_module_level_read_needs_the_module_marked() -> None:
    planted = 'import os\nimport pytest\n_TOKEN = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")\n'
    assert _unmarked_secret_tests(ast.parse(planted)) == ["module level, line 3"]
    marked = planted + "pytestmark = [pytest.mark.live, pytest.mark.secrets]\n"
    assert _unmarked_secret_tests(ast.parse(marked)) == []


def test_the_walker_sees_the_known_credentialed_tests() -> None:
    """The control on the tree: the known credentialed modules are found, so a broken detector can't pass."""
    known = {
        "packages/datasources/tests/integration/test_redshift_driver_live.py",
        "packages/datasources/tests/integration/test_transaction_live.py",
        "packages/models/tests/live/test_claude_cli_structured_output_live.py",
        "packages/search/tests/test_embedded_smoke.py",
    }
    seen: set[str] = set()
    for path in _test_files():
        tree = parse_python_path(path)
        # strip every secrets mark, then ask the walker what would be unmarked
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node.decorator_list = [d for d in node.decorator_list if not _secrets_mark(d)]
            if isinstance(node, (ast.Module, ast.ClassDef)):
                node.body = [s for s in node.body if not (isinstance(s, ast.Assign) and _secrets_mark(s.value))]
        if _unmarked_secret_tests(tree):
            seen.add(str(path.relative_to(_REPO_ROOT)))
    assert known <= seen, known - seen
