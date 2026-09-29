"""every live Redshift test reaches the warehouse only through the one-login gate.

the live tests log in as a production warehouse user that Redshift locks after
five failed logins. the gate (``tests/unit/_helpers/redshift_live_gate.py``)
sends the password once per session, and that holds only while:

- the ``redshift_config`` fixture that runs it is session-scoped -- pytest runs
  a session fixture once and re-raises its failure to every later request; at
  any narrower scope the gate would log in again per test;
- no other module defines a fixture by that name, which would shadow the gated
  one; and
- no live test builds a Redshift connection of its own, calls
  ``central_reporting()`` directly, or imports ``redshift_connector`` -- each
  logs in without the gate.

it walks every module under the integration directory, subdirectories
included.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_INTEGRATION = Path(__file__).resolve().parents[1] / "integration"
_CONFTEST = _INTEGRATION / "conftest.py"
_FIXTURE_NAMES = frozenset({"redshift_config", "redshift_creds"})


def _is_pytest_fixture(node: ast.expr) -> bool:
    """whether ``node`` is ``pytest.fixture``.

    :param node: a decorator expression, or the callee of one
    :ptype node: ast.expr
    :return: True for ``pytest.fixture``
    :rtype: bool
    """
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "fixture"
        and isinstance(node.value, ast.Name)
        and node.value.id == "pytest"
    )


def _fixture_scope(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """the scope a ``pytest.fixture`` decorator gives ``func``.

    :param func: a function definition
    :ptype func: ast.FunctionDef | ast.AsyncFunctionDef
    :return: the declared scope, ``"function"`` when none is declared, None when not a fixture
    :rtype: str | None
    """
    scope: str | None = None
    for decorator in func.decorator_list:
        if _is_pytest_fixture(decorator):
            scope = "function"
        elif isinstance(decorator, ast.Call) and _is_pytest_fixture(decorator.func):
            scope = "function"
            for keyword in decorator.keywords:
                if keyword.arg == "scope" and isinstance(keyword.value, ast.Constant):
                    scope = str(keyword.value.value)
    return scope


def _called_name(func: ast.expr) -> str:
    """the dotted name a call targets, or an empty string when it is not a plain name.

    :param func: the callee of a call
    :ptype func: ast.expr
    :return: e.g. ``RedshiftConnectionConfig`` or ``RedshiftConnectionConfig.model_validate``
    :rtype: str
    """
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


#: the fields that say WHICH warehouse user logs in; a live test that sets any of them logs in with a
#: login the gate never checked.
_IDENTITY_FIELDS = frozenset({"host", "port", "database", "username", "password_ref"})


def _sets_an_identity_field(node: ast.expr) -> bool:
    """whether ``node`` is a dict literal with an identity key.

    :param node: an expression
    :ptype node: ast.expr
    :return: True for ``{..., "host": ..., ...}`` or any other key in :data:`_IDENTITY_FIELDS`
    :rtype: bool
    """
    return isinstance(node, ast.Dict) and any(
        isinstance(key, ast.Constant) and key.value in _IDENTITY_FIELDS for key in node.keys
    )


def _bypass(node: ast.AST) -> str | None:
    """the rule ``node`` breaks by reaching the warehouse without the gate, or None.

    :param node: any node of a live test module
    :ptype node: ast.AST
    :return: a short name for the broken rule, or None when the node is fine
    :rtype: str | None
    """
    rule: str | None = None
    if isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "redshift_connector" for alias in node.names):
        rule = "imports redshift_connector"
    elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "redshift_connector":
        rule = "imports from redshift_connector"
    elif isinstance(node, ast.Call):
        name = _called_name(node.func)
        if name == "central_reporting":
            rule = "calls central_reporting() instead of taking redshift_config"
        elif name == "RedshiftConnectionConfig" and any(keyword.arg in _IDENTITY_FIELDS for keyword in node.keywords):
            rule = "constructs a connection with its own identity"
        elif name == "RedshiftConnectionConfig.model_validate" and any(
            _sets_an_identity_field(arg) for arg in node.args
        ):
            rule = "validates a connection with its own identity"
        elif name.endswith(".model_copy") and any(
            keyword.arg == "update" and _sets_an_identity_field(keyword.value) for keyword in node.keywords
        ):
            rule = "copies a connection onto another identity"
    return rule


def _live_modules() -> list[Path]:
    """every module under the integration directory, subdirectories included, but its conftest.

    :return: the test modules
    :rtype: list[Path]
    """
    return sorted(path for path in _INTEGRATION.rglob("*.py") if path != _CONFTEST)


def _where(path: Path, node: ast.AST) -> str:
    """a module and line, relative to the integration directory so two same-named modules differ.

    :param path: the module
    :ptype path: Path
    :param node: the node in it
    :ptype node: ast.AST
    :return: e.g. ``sub/test_x.py:12``
    :rtype: str
    """
    return f"{path.relative_to(_INTEGRATION)}:{getattr(node, 'lineno', 0)}"


def test_the_gate_fixture_is_session_scoped() -> None:
    tree = ast.parse(_CONFTEST.read_text())
    scopes = {
        node.name: _fixture_scope(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    assert scopes.get("redshift_config") == "session", (
        f"{_CONFTEST.name} must define redshift_config with scope='session'; at any narrower scope "
        f"the gate logs in again for every test (found {scopes.get('redshift_config')!r})"
    )


def test_no_live_module_defines_its_own_credentials_fixture() -> None:
    offenders = [
        f"{_where(path, node)} {node.name}"
        for path in _live_modules()
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in _FIXTURE_NAMES
    ]

    assert offenders == [], f"these shadow the gated fixture in {_CONFTEST.name}: {offenders}"


def test_no_live_test_logs_in_without_the_gate() -> None:
    offenders = [
        f"{_where(path, node)} {rule}"
        for path in _live_modules()
        for node in ast.walk(ast.parse(path.read_text()))
        if (rule := _bypass(node)) is not None
    ]

    assert offenders == [], (
        f"derive each connection from the redshift_config fixture instead; these reach the "
        f"warehouse without the one-login gate: {offenders}"
    )


# ---------------------------------------------------------------------------
# the detectors, against code they must catch and code they must pass
# ---------------------------------------------------------------------------

#: the login-identifying fields, spelled out rather than read from :data:`_IDENTITY_FIELDS`, so
#: narrowing that set fails these tests instead of narrowing them with it.
_SPELLED_OUT_IDENTITY = ("host", "port", "database", "username", "password_ref")

#: one snippet per way a live test can reach the warehouse without the gate.
_BYPASSES = [
    "import redshift_connector",
    "import redshift_connector.core",
    "from redshift_connector import connect",
    "central_reporting()",
    *(f"RedshiftConnectionConfig({field}=value)" for field in _SPELLED_OUT_IDENTITY),
    *(f"RedshiftConnectionConfig.model_validate({{'{field}': value}})" for field in _SPELLED_OUT_IDENTITY),
    *(f"base.model_copy(update={{'{field}': value}})" for field in _SPELLED_OUT_IDENTITY),
]

#: what the live tests do today, which must stay allowed.
_GATED = [
    "import redshift_connector_helpers",
    "RedshiftConnectionConfig.model_validate({**base.model_dump(exclude_unset=True), 'executor_max_workers': 4})",
    "base.model_copy(update={'query_timeout_seconds': 60})",
    "RedshiftDriver(config, datasource_name='central-reporting')",
]


def _flagged(snippet: str) -> list[str]:
    """every rule the detectors report for ``snippet``.

    :param snippet: python source
    :ptype snippet: str
    :return: the rules broken, empty when none
    :rtype: list[str]
    """
    return [rule for node in ast.walk(ast.parse(snippet)) if (rule := _bypass(node)) is not None]


@pytest.mark.parametrize("snippet", _BYPASSES)
def test_the_detector_catches_each_bypass(snippet: str) -> None:
    assert _flagged(snippet), f"the gate check would let this through: {snippet}"


@pytest.mark.parametrize("snippet", _GATED)
def test_the_detector_passes_the_gated_pattern(snippet: str) -> None:
    assert _flagged(snippet) == [], f"the gate check would refuse the gated pattern: {snippet}"


def test_the_scope_reader_sees_session_and_narrower_scopes() -> None:
    def scope_of(source: str) -> str | None:
        func = ast.parse(source).body[0]
        assert isinstance(func, ast.FunctionDef)
        return _fixture_scope(func)

    assert scope_of("@pytest.fixture(scope='session')\ndef f(): pass") == "session"
    assert scope_of("@pytest.fixture(scope='module')\ndef f(): pass") == "module"
    assert scope_of("@pytest.fixture\ndef f(): pass") == "function"
    assert scope_of("def f(): pass") is None
