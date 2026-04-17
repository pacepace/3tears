"""tests for :mod:`threetears.agent.workspace.validators`.

covers :func:`dispatch_validators` (pattern matching, multi-validator
ordering, WVE passthrough, non-WVE wrapping) and
:func:`_resolve_validator` (caching, bad dotted paths, non-callable
target, missing module). an integration test wires
:func:`_write_file_atomic` to a fake pool + validator to verify that a
validation failure aborts the transaction before any INSERT/UPSERT fires.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.workspace.config import ValidatorEntry
from threetears.agent.workspace.tools.helpers import _write_file_atomic
from threetears.agent.workspace.validators import (
    WorkspaceValidationError,
    _resolve_validator,
    dispatch_validators,
)
from threetears.agent.workspace import validators as validators_module


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    id: UUID
    name: str = "ws"


@dataclass
class _FakeTransaction:
    parent: _FakeConnection
    entered: bool = False
    exited: bool = False

    async def __aenter__(self) -> _FakeTransaction:
        self.entered = True
        self.parent.transaction_open = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.exited = True
        self.parent.transaction_open = False


@dataclass
class _FakeConnection:
    head_row: dict[str, Any] | None = None
    journal_max_version: int = 0
    executions: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    fetchrows: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False

    def transaction(self) -> _FakeTransaction:
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args, self.transaction_open))
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """dispatch by SQL shape: journal-max SELECT returns a row with
        ``max_version``; head SELECT (and any fallback) returns ``head_row``.
        """
        self.fetchrows.append((query, args, self.transaction_open))
        result: dict[str, Any] | None
        if "COALESCE(MAX(version)" in query:
            result = {"max_version": self.journal_max_version}
        else:
            result = self.head_row
        return result


@dataclass
class _FakeAcquireCM:
    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool:
    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=self.conn)


# ---------------------------------------------------------------------------
# dynamic test-only module for validator resolution
# ---------------------------------------------------------------------------


def _install_stub_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    attrs: dict[str, Any],
) -> types.ModuleType:
    """register a throwaway module under ``module_name`` for this test only.

    returns the created module so the test can mutate it mid-flight (e.g.
    simulate module reload between resolver calls).
    """
    module = types.ModuleType(module_name)
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


@pytest.fixture(autouse=True)
def _clear_resolve_cache() -> None:
    """every test runs against a clean _RESOLVED cache so order doesn't leak."""
    validators_module._RESOLVED.clear()


# ---------------------------------------------------------------------------
# dispatch_validators: basic paths
# ---------------------------------------------------------------------------


def test_dispatch_validators_empty_list_is_noop() -> None:
    """no validators -> never imports anything, never raises."""
    dispatch_validators([], "anything.yaml", b"payload")


def test_dispatch_validators_no_matching_pattern_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """no pattern matches relative_path -> no resolver calls, no fn calls."""
    calls: list[tuple[str, Any]] = []

    def _never(relpath: str, content: Any) -> None:
        calls.append((relpath, content))

    _install_stub_module(monkeypatch, "tests_validators_stub_a", {"never": _never})
    entries = [
        ValidatorEntry(pattern="*.json", validator="tests_validators_stub_a.never"),
    ]
    dispatch_validators(entries, "settings.yaml", b"{}")
    assert calls == []


def test_dispatch_validators_single_matching_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """exactly one matching pattern -> validator fired exactly once with args."""
    calls: list[tuple[str, Any]] = []

    def _ok(relpath: str, content: Any) -> None:
        calls.append((relpath, content))

    _install_stub_module(monkeypatch, "tests_validators_stub_b", {"ok": _ok})
    entries = [
        ValidatorEntry(pattern="*.yaml", validator="tests_validators_stub_b.ok"),
    ]
    dispatch_validators(entries, "config.yaml", b"hello")
    assert calls == [("config.yaml", b"hello")]


def test_dispatch_validators_multiple_matching_all_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """two matching patterns -> both validators fired, in list order."""
    order: list[str] = []

    def _first(relpath: str, content: Any) -> None:
        order.append("first")

    def _second(relpath: str, content: Any) -> None:
        order.append("second")

    _install_stub_module(
        monkeypatch,
        "tests_validators_stub_c",
        {"first": _first, "second": _second},
    )
    entries = [
        ValidatorEntry(pattern="*.yaml", validator="tests_validators_stub_c.first"),
        ValidatorEntry(pattern="*.yaml", validator="tests_validators_stub_c.second"),
    ]
    dispatch_validators(entries, "a.yaml", b"x")
    assert order == ["first", "second"]


def test_dispatch_validators_first_failure_aborts_later_validators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WVE on the first matching validator -> second validator never runs."""
    order: list[str] = []

    def _boom(relpath: str, content: Any) -> None:
        order.append("boom")
        raise WorkspaceValidationError(pattern="*.yaml", validator_path="ignored", reason="nope")

    def _later(relpath: str, content: Any) -> None:
        order.append("later")

    _install_stub_module(
        monkeypatch,
        "tests_validators_stub_d",
        {"boom": _boom, "later": _later},
    )
    entries = [
        ValidatorEntry(pattern="*.yaml", validator="tests_validators_stub_d.boom"),
        ValidatorEntry(pattern="*.yaml", validator="tests_validators_stub_d.later"),
    ]
    with pytest.raises(WorkspaceValidationError) as excinfo:
        dispatch_validators(entries, "a.yaml", b"x")
    assert order == ["boom"]
    assert excinfo.value.reason == "nope"


def test_dispatch_validators_wve_from_validator_is_reraised_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validator raising WVE -> same instance propagates, not re-wrapped."""
    sentinel = WorkspaceValidationError(pattern="kept", validator_path="kept", reason="kept")

    def _raise_wve(relpath: str, content: Any) -> None:
        raise sentinel

    _install_stub_module(
        monkeypatch,
        "tests_validators_stub_e",
        {"raise_wve": _raise_wve},
    )
    entries = [
        ValidatorEntry(pattern="*.yaml", validator="tests_validators_stub_e.raise_wve"),
    ]
    with pytest.raises(WorkspaceValidationError) as excinfo:
        dispatch_validators(entries, "a.yaml", b"x")
    assert excinfo.value is sentinel


def test_dispatch_validators_non_wve_gets_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validator raising ValueError -> wrapped into WVE w/ pattern+path+reason."""

    def _raise_value(relpath: str, content: Any) -> None:
        raise ValueError("bad")

    _install_stub_module(
        monkeypatch,
        "tests_validators_stub_f",
        {"raise_value": _raise_value},
    )
    entries = [
        ValidatorEntry(
            pattern="*.yaml",
            validator="tests_validators_stub_f.raise_value",
        ),
    ]
    with pytest.raises(WorkspaceValidationError) as excinfo:
        dispatch_validators(entries, "a.yaml", b"x")
    assert excinfo.value.pattern == "*.yaml"
    assert excinfo.value.validator_path == "tests_validators_stub_f.raise_value"
    assert excinfo.value.reason == "bad"
    assert isinstance(excinfo.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# _resolve_validator: caching + error paths
# ---------------------------------------------------------------------------


def test_resolve_validator_caches_callable_object_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """same dotted path resolved twice -> identical callable object (one import)."""

    def _fn(relpath: str, content: Any) -> None:
        return None

    module = _install_stub_module(monkeypatch, "tests_validators_stub_g", {"fn": _fn})
    first = _resolve_validator("tests_validators_stub_g.fn")
    # replace the module's attribute AFTER first resolve; a cached
    # callable must still return the original closure.
    module.fn = lambda *a, **k: None  # type: ignore[assignment]
    second = _resolve_validator("tests_validators_stub_g.fn")
    assert first is second
    assert first is _fn


def test_resolve_validator_no_dot_raises_value_error() -> None:
    """bare name with no dotted prefix -> ValueError at resolve time."""
    with pytest.raises(ValueError) as excinfo:
        _resolve_validator("no_dot")
    assert "dotted import path" in str(excinfo.value)


def test_resolve_validator_non_callable_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dotted path pointing at a non-callable attribute -> ValueError."""
    _install_stub_module(
        monkeypatch,
        "tests_validators_stub_h",
        {"not_a_fn": 42},
    )
    with pytest.raises(ValueError) as excinfo:
        _resolve_validator("tests_validators_stub_h.not_a_fn")
    assert "not callable" in str(excinfo.value)


def test_resolve_validator_missing_module_propagates_import_error() -> None:
    """unknown module path -> ImportError propagates (config bug, fail loud)."""
    with pytest.raises(ImportError):
        _resolve_validator("tests_validators_does_not_exist_ever.anything")


def test_resolve_validator_missing_attr_propagates_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """module exists but lacks the attr -> AttributeError propagates."""
    _install_stub_module(monkeypatch, "tests_validators_stub_i", {"present": 1})
    with pytest.raises(AttributeError):
        _resolve_validator("tests_validators_stub_i.missing")


# ---------------------------------------------------------------------------
# integration: _write_file_atomic + validator rolls back before INSERT/UPSERT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file_atomic_validator_failure_aborts_before_inserts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validator raises -> WVE, no INSERT/UPSERT, transaction exit recorded."""

    def _reject(relpath: str, content: Any) -> None:
        raise ValueError("schema broken")

    _install_stub_module(
        monkeypatch,
        "tests_validators_stub_int",
        {"reject": _reject},
    )
    entries = [
        ValidatorEntry(
            pattern="*.yaml",
            validator="tests_validators_stub_int.reject",
        ),
    ]
    pool = _FakePool()
    ws = _FakeWorkspaceEntity(id=uuid4())
    with pytest.raises(WorkspaceValidationError) as excinfo:
        await _write_file_atomic(
            db_pool=pool,
            workspace=ws,
            relative_path="settings.yaml",
            content=b"audience_units: not-a-list",
            action="update",
            actor_id=uuid4(),
            correlation_id=uuid4(),
            expected_sha256=None,
            workspace_file_collection=None,
            workspace_file_version_collection=None,
            workspace_collection=None,
            validators=entries,
        )
    assert excinfo.value.pattern == "*.yaml"
    assert excinfo.value.reason == "schema broken"
    # fetchrow ran (head lookup inside tx), but no INSERT/UPSERT ran.
    assert len(pool.conn.fetchrows) == 1
    assert pool.conn.executions == []
    # transaction opened and closed -- asyncpg rolls back on exception.
    assert len(pool.conn.transactions) == 1
    assert pool.conn.transactions[0].entered is True
    assert pool.conn.transactions[0].exited is True


@pytest.mark.asyncio
async def test_write_file_atomic_validator_pass_proceeds_to_inserts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validator returns None -> normal three-row transaction happens."""
    hits: list[tuple[str, Any]] = []

    def _ok(relpath: str, content: Any) -> None:
        hits.append((relpath, content))

    _install_stub_module(
        monkeypatch,
        "tests_validators_stub_int_ok",
        {"ok": _ok},
    )
    entries = [
        ValidatorEntry(
            pattern="*.yaml",
            validator="tests_validators_stub_int_ok.ok",
        ),
    ]
    pool = _FakePool()
    ws = _FakeWorkspaceEntity(id=uuid4())
    new_version, _new_sha = await _write_file_atomic(
        db_pool=pool,
        workspace=ws,
        relative_path="settings.yaml",
        content=b"ok",
        action="create",
        actor_id=uuid4(),
        correlation_id=uuid4(),
        expected_sha256=None,
        workspace_file_collection=None,
        workspace_file_version_collection=None,
        workspace_collection=None,
        validators=entries,
    )
    assert new_version == 1
    assert hits == [("settings.yaml", b"ok")]
    # two fetchrows (head + journal-max) + 3 executes, all inside the transaction.
    assert len(pool.conn.fetchrows) == 2
    assert len(pool.conn.executions) == 3


@pytest.mark.asyncio
async def test_write_file_atomic_no_validators_is_noop() -> None:
    """validators=None preserves pre-shard behavior exactly."""
    pool = _FakePool()
    ws = _FakeWorkspaceEntity(id=uuid4())
    new_version, _new_sha = await _write_file_atomic(
        db_pool=pool,
        workspace=ws,
        relative_path="any.txt",
        content=b"hello",
        action="create",
        actor_id=uuid4(),
        correlation_id=uuid4(),
        expected_sha256=None,
        workspace_file_collection=None,
        workspace_file_version_collection=None,
        workspace_collection=None,
        validators=None,
    )
    assert new_version == 1
    assert len(pool.conn.executions) == 3


# ---------------------------------------------------------------------------
# glob semantics: full_match anchoring (mirrors sandbox contract)
# ---------------------------------------------------------------------------


def test_dispatch_validators_single_segment_pattern_does_not_match_deep_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``config/*.yaml`` must NOT match ``deep/nested/config/x.yaml``.

    under the old :meth:`PurePosixPath.match` semantics this would have
    matched (match compares only the right side). with
    :meth:`PurePosixPath.full_match` the pattern is anchored end-to-end,
    matching the sandbox allow-list contract.
    """
    calls: list[tuple[str, Any]] = []

    def _validator(relpath: str, content: Any) -> None:
        calls.append((relpath, content))

    _install_stub_module(monkeypatch, "tests_validators_glob_a", {"fn": _validator})
    entries = [
        ValidatorEntry(
            pattern="config/*.yaml",
            validator="tests_validators_glob_a.fn",
        ),
    ]
    dispatch_validators(entries, "deep/nested/config/x.yaml", b"x")
    assert calls == []


def test_dispatch_validators_single_segment_pattern_matches_anchored_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``config/*.yaml`` matches ``config/x.yaml`` (the anchored case)."""
    calls: list[tuple[str, Any]] = []

    def _validator(relpath: str, content: Any) -> None:
        calls.append((relpath, content))

    _install_stub_module(monkeypatch, "tests_validators_glob_b", {"fn": _validator})
    entries = [
        ValidatorEntry(
            pattern="config/*.yaml",
            validator="tests_validators_glob_b.fn",
        ),
    ]
    dispatch_validators(entries, "config/x.yaml", b"x")
    assert calls == [("config/x.yaml", b"x")]


def test_dispatch_validators_recursive_glob_matches_any_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``**/*.yaml`` matches arbitrary depth (recursive glob semantics)."""
    calls: list[tuple[str, Any]] = []

    def _validator(relpath: str, content: Any) -> None:
        calls.append((relpath, content))

    _install_stub_module(monkeypatch, "tests_validators_glob_c", {"fn": _validator})
    entries = [
        ValidatorEntry(
            pattern="**/*.yaml",
            validator="tests_validators_glob_c.fn",
        ),
    ]
    # top-level
    dispatch_validators(entries, "a.yaml", b"x")
    # nested
    dispatch_validators(entries, "deep/nested/x.yaml", b"y")
    assert calls == [("a.yaml", b"x"), ("deep/nested/x.yaml", b"y")]


# ---------------------------------------------------------------------------
# WorkspaceValidationError surface
# ---------------------------------------------------------------------------


def test_workspace_validation_error_exposes_attrs() -> None:
    """WVE subclass of ValueError carries pattern, validator_path, reason."""
    exc = WorkspaceValidationError(
        pattern="*/audience_settings.yaml",
        validator_path="mod.fn",
        reason="missing required field",
    )
    assert isinstance(exc, ValueError)
    assert exc.pattern == "*/audience_settings.yaml"
    assert exc.validator_path == "mod.fn"
    assert exc.reason == "missing required field"
    assert "audience_settings.yaml" in str(exc)
    assert "mod.fn" in str(exc)
    assert "missing required field" in str(exc)
