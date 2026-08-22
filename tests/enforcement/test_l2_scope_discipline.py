"""L2 key-scope discipline guard (coll-task-03).

Two rules, each pinning one decision that was reached on evidence and would otherwise be
undone silently by the next author -- neither failure shows up as a test failure anywhere
else, which is why they are enforced statically.

(a) **``kv_key_scope_for`` is the ONLY producer of a scope value.** A scope is the leading
    token of ``$KV.{bucket}.{scope}.{table}.{body}`` and therefore the isolation boundary
    between principals. The obvious way to add the next principal is to interpolate a name
    -- ``f"tools-{sanitize_segment(mcp_name)}"`` -- and the dots-to-dash sanitizer is
    ``value.replace(".", "-")``, non-injective, with the collapse already recorded in
    ``core/namespaces.py``'s own source. Two pods sharing a scope is precisely the outcome
    this work exists to prevent, and it would present as a cache that occasionally serves
    another tenant's row rather than as anything that looks like a bug in the derivation.

(b) **The backstop exception type never appears in an ``except`` clause beside ``KvError``.**
    Three of ``l2_key``'s four call sites sit inside ``except KvError`` handlers that degrade
    to a warning. Widening one of those handlers to catch the scope error as well -- which
    reads as defensive tidying -- restores exactly the silent degradation the distinct type
    was introduced to prevent: the fleet runs with L2 off and nothing says so.

Mode via ``L2_SCOPE_ENFORCEMENT_MODE`` (default ``strict``), mirroring the sibling guards.
Exemptions live in ``_l2_scope_exemptions.txt`` and require a specific ``# rationale:`` line.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from threetears.enforcement.common import (
    MODE_REPORT,
    MODE_STRICT,
    Violation,
    apply_exemptions,
    emit_report,
    find_local_src_roots,
    iter_python_files,
    parse_exemptions_with_rationale,
    parse_python_file,
    resolve_mode,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXEMPTIONS_PATH = _REPO_ROOT / "tests" / "enforcement" / "_l2_scope_exemptions.txt"
_MODE_ENV_VAR = "L2_SCOPE_ENFORCEMENT_MODE"

_SCOPE_PROVENANCE_CATEGORY = "l2_scope.derived_outside_the_helper"
_BACKSTOP_SWALLOW_CATEGORY = "l2_scope.backstop_caught_with_kv_error"

#: the sanctioned producer. a ``kv_key_scope=`` value that is a call to this name (however
#: it was imported) is the shape every wiring site is expected to take.
_SCOPE_PRODUCER = "kv_key_scope_for"

#: the keyword every registry wiring site passes a scope through.
_SCOPE_KEYWORD = "kv_key_scope"

#: assignment targets treated as "this variable holds a scope", so the two-step evasion
#: (build it into a local, then pass the local) is caught at the build rather than the pass.
_SCOPE_NAME_FRAGMENT = "kv_key_scope"

#: the exception type ``l2_key`` raises as its backstop, plus its base. neither may be
#: caught alongside ``KvError``.
_BACKSTOP_NAMES: frozenset[str] = frozenset({"L2ScopeNotConfiguredError", "L2ScopeError"})

#: the transport error whose handlers degrade to a warning.
_DEGRADING_ERROR = "KvError"


# ---------------------------------------------------------------------------
# rule (a) -- only kv_key_scope_for produces a scope
# ---------------------------------------------------------------------------


def _called_name(func: ast.expr) -> str | None:
    """the bare callee name of a call target, however it was imported.

    :param func: the ``ast.Call.func`` node under inspection
    :ptype func: ast.expr
    :return: the callee's final name segment, or ``None`` when it is not a plain reference
    :rtype: str | None
    """
    result: str | None = None
    if isinstance(func, ast.Name):
        result = func.id
    elif isinstance(func, ast.Attribute):
        result = func.attr
    return result


def _is_sanctioned_scope_expression(value: ast.expr) -> bool:
    """whether an expression is an acceptable source for a scope value.

    Acceptable: a call to :data:`_SCOPE_PRODUCER`, ``None`` (the "leave unchanged" argument),
    or a plain name / attribute reference (a value produced elsewhere and threaded through --
    rule (a)'s second half catches the case where that elsewhere built it by hand).

    Rejected: f-strings, concatenation, and any OTHER call -- the shapes a name-derived scope
    actually takes.

    :param value: the expression supplied as the scope
    :ptype value: ast.expr
    :return: whether the expression is a sanctioned source
    :rtype: bool
    """
    result = False
    if isinstance(value, ast.Call):
        result = _called_name(value.func) == _SCOPE_PRODUCER
    elif isinstance(value, ast.Constant) and value.value is None:
        result = True
    elif isinstance(value, ast.Name | ast.Attribute):
        result = True
    return result


def _is_hand_built(value: ast.expr) -> bool:
    """whether an expression hand-builds a string rather than naming a produced one.

    :param value: the expression assigned to a scope-named variable
    :ptype value: ast.expr
    :return: whether the expression interpolates, concatenates, or calls something else
    :rtype: bool
    """
    result = False
    if isinstance(value, ast.JoinedStr | ast.BinOp):
        result = True
    elif isinstance(value, ast.Call):
        result = _called_name(value.func) != _SCOPE_PRODUCER
    return result


def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """the flat names an assignment writes to.

    :param node: assignment node
    :ptype node: ast.Assign | ast.AnnAssign
    :return: target names, attribute names included
    :rtype: list[str]
    """
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return names


def find_hand_derived_scopes(scan_roots: tuple[Path, ...]) -> list[Violation]:
    """flag every scope value not produced by :data:`_SCOPE_PRODUCER`.

    Two shapes: a ``kv_key_scope=`` keyword whose value is built inline, and an assignment
    to a scope-named variable whose value is built by hand (the same thing, one statement
    earlier).

    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg != _SCOPE_KEYWORD:
                            continue
                        if _is_sanctioned_scope_expression(keyword.value):
                            continue
                        violations.append(
                            Violation(
                                category=_SCOPE_PROVENANCE_CATEGORY,
                                file=source,
                                line=node.lineno,
                                symbol=_SCOPE_KEYWORD,
                                reason=(
                                    f"{_SCOPE_KEYWORD}= is built inline; a scope must come from "
                                    f"{_SCOPE_PRODUCER}(), never from a sanitized display name "
                                    "(the dots-to-dash sanitizer is non-injective, so two "
                                    "principals can collapse onto one scope)"
                                ),
                            )
                        )
                if isinstance(node, ast.Assign | ast.AnnAssign):
                    if node.value is None:
                        continue
                    named = [n for n in _assignment_targets(node) if _SCOPE_NAME_FRAGMENT in n]
                    if not named or not _is_hand_built(node.value):
                        continue
                    violations.append(
                        Violation(
                            category=_SCOPE_PROVENANCE_CATEGORY,
                            file=source,
                            line=node.lineno,
                            symbol=named[0],
                            reason=(
                                f"'{named[0]}' is assigned a hand-built value; a scope must come "
                                f"from {_SCOPE_PRODUCER}()"
                            ),
                        )
                    )
    return violations


# ---------------------------------------------------------------------------
# rule (b) -- the backstop is never caught beside KvError
# ---------------------------------------------------------------------------


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    """the exception type names one ``except`` clause catches.

    :param handler: the except-handler node
    :ptype handler: ast.ExceptHandler
    :return: caught type names, attribute-qualified ones reduced to their final segment
    :rtype: set[str]
    """
    if handler.type is None:
        return set()
    candidates = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    names: set[str] = set()
    for candidate in candidates:
        name = _called_name(candidate)
        if name is not None:
            names.add(name)
    return names


def find_backstop_caught_with_kv_error(scan_roots: tuple[Path, ...]) -> list[Violation]:
    """flag any ``except`` clause catching the backstop type alongside ``KvError``.

    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                caught = _handler_names(node)
                shared = caught & _BACKSTOP_NAMES
                if not shared or _DEGRADING_ERROR not in caught:
                    continue
                violations.append(
                    Violation(
                        category=_BACKSTOP_SWALLOW_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=sorted(shared)[0],
                        reason=(
                            f"'{sorted(shared)[0]}' is caught alongside {_DEGRADING_ERROR}; those "
                            "handlers degrade to a warning, so the missing scope would be "
                            "swallowed and the fleet would run with L2 silently off"
                        ),
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _assert_clean(violations: list[Violation], domain: str) -> None:
    """apply exemptions + mode and fail in strict mode with a rendered report.

    :param violations: raw walker output
    :ptype violations: list[Violation]
    :param domain: report domain label
    :ptype domain: str
    :return: nothing
    :rtype: None
    :raises pytest.fail.Exception: in strict mode with surviving violations
    """
    exemptions = parse_exemptions_with_rationale(_EXEMPTIONS_PATH)
    filtered = apply_exemptions(violations, exemptions, _REPO_ROOT)
    mode = resolve_mode(_MODE_ENV_VAR, default=MODE_STRICT)
    report = emit_report(filtered, (_REPO_ROOT,), exemptions, mode, _REPO_ROOT, domain=domain)
    print(report, file=sys.stderr)
    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"{domain} found {len(filtered)} violation(s):\n{report}")


class TestL2ScopeDiscipline:
    """the two scope rules over the 3tears package tree."""

    def test_every_scope_comes_from_the_helper(self) -> None:
        """no scope value is hand-built from a name."""
        violations = find_hand_derived_scopes(find_local_src_roots(_REPO_ROOT))
        _assert_clean(violations, _SCOPE_PROVENANCE_CATEGORY)

    def test_the_backstop_is_never_caught_with_kv_error(self) -> None:
        """no handler degrades a missing scope to a warning."""
        violations = find_backstop_caught_with_kv_error(find_local_src_roots(_REPO_ROOT))
        _assert_clean(violations, _BACKSTOP_SWALLOW_CATEGORY)


class TestWalkersFlagPlantedViolations:
    """each walker must fire on a planted violation (self-test of the guard).

    A guard whose walker silently matches nothing reads exactly like a guard with nothing
    to report, so each rule is exercised against source that SHOULD trip it -- and against
    source that should not, since a walker that flags everything is equally useless.
    """

    def test_scope_walker_flags_an_interpolated_scope(self, tmp_path: Path) -> None:
        source = (
            "from threetears.core.namespaces import sanitize_segment\n\n\n"
            "def wire(registry, mcp_name):\n"
            '    registry.configure(kv_key_scope=f"tools-{sanitize_segment(mcp_name)}")\n'
        )
        (tmp_path / "planted_interpolated.py").write_text(source, encoding="utf-8")

        violations = find_hand_derived_scopes((tmp_path,))

        assert [v.symbol for v in violations] == ["kv_key_scope"]

    def test_scope_walker_flags_a_hand_built_local(self, tmp_path: Path) -> None:
        source = (
            "from threetears.core.namespaces import sanitize_segment\n\n\n"
            "def wire(registry, mcp_name):\n"
            "    kv_key_scope = sanitize_segment(mcp_name)\n"
            "    registry.configure(kv_key_scope=kv_key_scope)\n"
        )
        (tmp_path / "planted_local.py").write_text(source, encoding="utf-8")

        violations = find_hand_derived_scopes((tmp_path,))

        assert [v.symbol for v in violations] == ["kv_key_scope"]

    def test_scope_walker_accepts_the_helper(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats import Principal, kv_key_scope_for\n\n\n"
            "def wire(registry, agent_id):\n"
            "    scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=agent_id)\n"
            "    registry.configure(kv_key_scope=scope)\n"
            "    registry.configure(kv_key_scope=kv_key_scope_for(Principal.HUB))\n"
        )
        (tmp_path / "sanctioned.py").write_text(source, encoding="utf-8")

        assert find_hand_derived_scopes((tmp_path,)) == []

    def test_backstop_walker_flags_a_widened_handler(self, tmp_path: Path) -> None:
        source = (
            "from threetears.core.exceptions import L2ScopeNotConfiguredError\n"
            "from threetears.nats.errors import KvError\n\n\n"
            "async def read(self, entity_id):\n"
            "    try:\n"
            "        return await self._kv.get(key=self.l2_key(entity_id))\n"
            "    except (KvError, L2ScopeNotConfiguredError):\n"
            "        return None\n"
        )
        (tmp_path / "planted_handler.py").write_text(source, encoding="utf-8")

        violations = find_backstop_caught_with_kv_error((tmp_path,))

        assert [v.symbol for v in violations] == ["L2ScopeNotConfiguredError"]

    def test_backstop_walker_accepts_a_kv_error_only_handler(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.errors import KvError\n\n\n"
            "async def read(self, entity_id):\n"
            "    try:\n"
            "        return await self._kv.get(key=self.l2_key(entity_id))\n"
            "    except KvError:\n"
            "        return None\n"
        )
        (tmp_path / "sanctioned_handler.py").write_text(source, encoding="utf-8")

        assert find_backstop_caught_with_kv_error((tmp_path,)) == []
