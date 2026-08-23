"""KV grant-capability discipline guard (coll-task-05a).

Three rules over :mod:`threetears.nats.subject_permissions`, each pinning a decision whose
violation is SILENT at every other layer. A NATS grant that is too wide raises nothing and passes
every test; a grant that is too narrow raises nothing either, because a refused JetStream request is
never answered and arrives as a ten-second deadline that reads as an unreachable broker. Neither
direction shows up as a test failure anywhere else, which is why they are enforced statically.

(a) **``declare`` belongs to the declaring identity alone.** ``JsCapability.KV_SCOPED_DECLARE``
    carries ``STREAM.CREATE`` and ``STREAM.UPDATE``. On a stream one principal owns those are
    ordinary lifecycle verbs; on the SHARED ``{ns}-collections`` stream ``UPDATE`` is a read-all
    primitive, because ``republish`` and ``sources`` mirror every key -- every principal's -- onto a
    subject the caller names. So the capability is bound to the hub's bootstrap declaration and to
    nothing else, and a pod principal may never hold it.

(b) **The shared bucket is never granted unscoped.** Adding ``{ns}-collections`` to a new principal
    and forgetting the scope silently restores the firehose the scoping exists to close: the grant
    still works, so nothing fails, and every key in the platform is readable and writable by that
    principal again.

(c) **Every bucket declares its scope and write intent explicitly.** ``JsResource.kv`` takes both as
    keyword-only parameters with no defaults, so the language already refuses an omission -- this
    walker covers the way around that: constructing :class:`JsResource` directly, or reintroducing a
    default. A new bucket must be a decision about both, not an inheritance of somebody else's.

Mode via ``KV_GRANT_ENFORCEMENT_MODE`` (default ``strict``), mirroring the sibling guards.
Exemptions live in ``_kv_grant_exemptions.txt`` and require a specific ``# rationale:`` line.
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
_EXEMPTIONS_PATH = _REPO_ROOT / "tests" / "enforcement" / "_kv_grant_exemptions.txt"
_MODE_ENV_VAR = "KV_GRANT_ENFORCEMENT_MODE"

_DECLARE_CATEGORY = "kv_grant.declare_outside_the_declarer"
_UNSCOPED_SHARED_CATEGORY = "kv_grant.shared_bucket_granted_unscoped"
_IMPLICIT_DECISION_CATEGORY = "kv_grant.bucket_without_an_explicit_decision"

#: the resource-record constructors this guard reads.
_KV_FACTORY = "kv"
_RESOURCE_CLASS = "JsResource"

#: the capability that carries CREATE + UPDATE, in both the shapes source can name it.
_DECLARE_CAPABILITY = "KV_SCOPED_DECLARE"

#: the ONLY function permitted to ask for it. ``coll-task-04a`` makes hub bootstrap the canonical
#: declarer of the collections bucket, so the hub's own resolver is the one place CREATE + UPDATE
#: is legitimate. Widening this set is a security decision, not a formatting one.
_SANCTIONED_DECLARERS: frozenset[str] = frozenset({"_hub"})

#: the bucket-name fragment that identifies the shared, key-scoped bucket. Matched on the FRAGMENT
#: rather than the whole name because the name is built as ``f"{ns}-collections"`` at every site.
_SHARED_BUCKET_FRAGMENT = "-collections"

#: the two decisions every bucket declaration must state.
_REQUIRED_KEYWORDS: tuple[str, ...] = ("scope", "writable")


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


def _is_kv_factory(node: ast.Call) -> bool:
    """whether ``node`` is a ``JsResource.kv(...)`` call.

    :param node: the call under inspection
    :ptype node: ast.Call
    :return: whether the call constructs a KV bucket resource through the factory
    :rtype: bool
    """
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == _KV_FACTORY
        and _called_name(node.func.value) == _RESOURCE_CLASS
    )


def _is_resource_constructor(node: ast.Call) -> bool:
    """whether ``node`` constructs :class:`JsResource` directly rather than through a factory.

    :param node: the call under inspection
    :ptype node: ast.Call
    :return: whether the call is a bare ``JsResource(...)``
    :rtype: bool
    """
    return isinstance(node.func, ast.Name) and node.func.id == _RESOURCE_CLASS


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    """the expression passed for keyword ``name``, or ``None`` when it is absent.

    :param node: the call under inspection
    :ptype node: ast.Call
    :param name: the keyword to look for
    :ptype name: str
    :return: the argument expression, or ``None``
    :rtype: ast.expr | None
    """
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _is_truthy_constant(value: ast.expr | None) -> bool:
    """whether ``value`` is a literal ``True``.

    :param value: the expression to test
    :ptype value: ast.expr | None
    :return: whether the expression is the constant ``True``
    :rtype: bool
    """
    return isinstance(value, ast.Constant) and value.value is True


def _names_declare_capability(value: ast.expr | None) -> bool:
    """whether ``value`` names :attr:`JsCapability.KV_SCOPED_DECLARE`.

    :param value: the expression to test
    :ptype value: ast.expr | None
    :return: whether the expression references the declaring capability
    :rtype: bool
    """
    return value is not None and _called_name(value) == _DECLARE_CAPABILITY


def _mentions_shared_bucket(value: ast.expr | None) -> bool:
    """whether ``value`` builds the shared collections bucket's name.

    Covers the plain literal and the ``f"{ns}-collections"`` form every resolver actually uses.

    :param value: the expression supplied as the bucket name
    :ptype value: ast.expr | None
    :return: whether the expression names the shared bucket
    :rtype: bool
    """
    result = False
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        result = _SHARED_BUCKET_FRAGMENT in value.value
    elif isinstance(value, ast.JoinedStr):
        result = any(
            isinstance(part, ast.Constant) and isinstance(part.value, str) and _SHARED_BUCKET_FRAGMENT in part.value
            for part in value.values
        )
    return result


def _bucket_name_expression(node: ast.Call) -> ast.expr | None:
    """the expression naming the bucket, from either the positional or the keyword form.

    :param node: a ``JsResource.kv(...)`` or ``JsResource(...)`` call
    :ptype node: ast.Call
    :return: the name expression, or ``None`` when the call names nothing
    :rtype: ast.expr | None
    """
    if node.args:
        return node.args[0]
    return _keyword(node, "name")


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    """map every node's line to the name of the function that contains it.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :return: line number -> enclosing function name
    :rtype: dict[int, str]
    """
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
            owner[line] = node.name
    return owner


# ---------------------------------------------------------------------------
# rule (a) -- declare belongs to the declaring identity alone
# ---------------------------------------------------------------------------


def find_declare_outside_the_declarer(
    scan_roots: tuple[Path, ...],
    sanctioned: frozenset[str],
) -> list[Violation]:
    """flag every request for the declaring capability outside a sanctioned declarer.

    Both shapes: ``JsResource.kv(..., declare=True)`` and any direct reference to
    ``JsCapability.KV_SCOPED_DECLARE``.

    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :param sanctioned: names of the functions permitted to declare
    :ptype sanctioned: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            tree = parse_python_file(source)
            if tree is None:
                continue
            owner = _enclosing_functions(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                asks = (_is_kv_factory(node) and _is_truthy_constant(_keyword(node, "declare"))) or (
                    _is_resource_constructor(node) and _names_declare_capability(_keyword(node, "capability"))
                )
                if not asks:
                    continue
                enclosing = owner.get(node.lineno, "<module>")
                if enclosing in sanctioned:
                    continue
                violations.append(
                    Violation(
                        category=_DECLARE_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=enclosing,
                        reason=(
                            f"'{enclosing}' asks for the declaring capability (STREAM.CREATE + "
                            "STREAM.UPDATE) on a shared stream; UPDATE sets republish/sources, which "
                            "mirror EVERY principal's keys onto a subject the caller names. it "
                            f"belongs to the declaring identity alone ({', '.join(sorted(sanctioned))})"
                        ),
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# rule (b) -- the shared bucket is never granted unscoped
# ---------------------------------------------------------------------------


def find_unscoped_shared_bucket_grants(scan_roots: tuple[Path, ...]) -> list[Violation]:
    """flag every collections-bucket declaration whose scope is absent or ``None``.

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
            owner = _enclosing_functions(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not (_is_kv_factory(node) or _is_resource_constructor(node)):
                    continue
                if not _mentions_shared_bucket(_bucket_name_expression(node)):
                    continue
                scope = _keyword(node, "scope")
                explicit = scope is not None and not (isinstance(scope, ast.Constant) and scope.value is None)
                if explicit:
                    continue
                violations.append(
                    Violation(
                        category=_UNSCOPED_SHARED_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=owner.get(node.lineno, "<module>"),
                        reason=(
                            "the shared collections bucket is declared without a key scope; the "
                            "grant would widen to $KV.{bucket}.> -- every principal's keys, read "
                            "and write -- and nothing would fail, because a grant that is too wide "
                            "works perfectly"
                        ),
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# rule (c) -- every bucket states both decisions
# ---------------------------------------------------------------------------


def find_implicit_bucket_decisions(scan_roots: tuple[Path, ...]) -> list[Violation]:
    """flag every bucket declaration that omits ``scope=`` or ``writable=``.

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
            owner = _enclosing_functions(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not (_is_kv_factory(node) or _is_resource_constructor(node)):
                    continue
                missing = [name for name in _REQUIRED_KEYWORDS if _keyword(node, name) is None]
                if not missing:
                    continue
                violations.append(
                    Violation(
                        category=_IMPLICIT_DECISION_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=owner.get(node.lineno, "<module>"),
                        reason=(
                            f"KV bucket declaration omits {', '.join(missing)}; a new bucket must "
                            "state its key scope and its write intent explicitly rather than "
                            "inherit another bucket's answer"
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


class TestKvGrantCapability:
    """the three capability rules over the 3tears package tree."""

    def test_only_the_declaring_identity_asks_to_declare(self) -> None:
        """no pod principal, and no other resolver, holds STREAM.CREATE + STREAM.UPDATE."""
        violations = find_declare_outside_the_declarer(find_local_src_roots(_REPO_ROOT), _SANCTIONED_DECLARERS)
        _assert_clean(violations, _DECLARE_CATEGORY)

    def test_the_shared_bucket_is_always_scoped(self) -> None:
        """no collections grant widens back to the whole bucket."""
        violations = find_unscoped_shared_bucket_grants(find_local_src_roots(_REPO_ROOT))
        _assert_clean(violations, _UNSCOPED_SHARED_CATEGORY)

    def test_every_bucket_states_its_scope_and_write_intent(self) -> None:
        """no bucket is added without a decision on both."""
        violations = find_implicit_bucket_decisions(find_local_src_roots(_REPO_ROOT))
        _assert_clean(violations, _IMPLICIT_DECISION_CATEGORY)

    def test_the_declare_capability_is_actually_held_by_the_hub(self) -> None:
        """the sanctioned declarer is a real site, not an empty allowance.

        A rule whose sanctioned set matches nothing passes trivially and would keep passing if the
        hub's declaration were deleted -- at which point a cold cluster cannot bootstrap the bucket
        and every pod's eager open fails on a config mismatch.
        """
        from threetears.nats.subject_permissions import Principal, build_permissions, capability_declares

        hub = build_permissions(Principal.HUB, conn_id="conn-1")
        declaring = [r.name for r in hub.js_resources if capability_declares(r.capability)]
        assert len(declaring) == 1, declaring
        assert _SHARED_BUCKET_FRAGMENT in declaring[0]


class TestWalkersFlagPlantedViolations:
    """each walker must fire on a planted violation and stay quiet on sanctioned source.

    A guard whose walker silently matches nothing reads exactly like a guard with nothing to
    report, and a walker that flags everything is equally useless -- so both directions are
    exercised.
    """

    def test_declare_walker_flags_a_pod_resolver(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.subject_permissions import JsResource\n\n\n"
            "def _agent_pod():\n"
            '    return (JsResource.kv("ns-collections", scope="s", writable=True, declare=True),)\n'
        )
        (tmp_path / "planted_declare.py").write_text(source, encoding="utf-8")

        violations = find_declare_outside_the_declarer((tmp_path,), _SANCTIONED_DECLARERS)

        assert [v.symbol for v in violations] == ["_agent_pod"]

    def test_declare_walker_flags_the_raw_constructor_too(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.subject_permissions import JsCapability, JsResource, JsResourceKind\n\n\n"
            "def _tool_pod():\n"
            "    return JsResource(\n"
            '        name="ns-collections",\n'
            "        kind=JsResourceKind.KV_BUCKET,\n"
            "        capability=JsCapability.KV_SCOPED_DECLARE,\n"
            '        scope="s",\n'
            "        writable=True,\n"
            "    )\n"
        )
        (tmp_path / "planted_raw_declare.py").write_text(source, encoding="utf-8")

        violations = find_declare_outside_the_declarer((tmp_path,), _SANCTIONED_DECLARERS)

        assert [v.symbol for v in violations] == ["_tool_pod"]

    def test_declare_walker_accepts_the_sanctioned_declarer(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.subject_permissions import JsResource\n\n\n"
            "def _hub():\n"
            '    return (JsResource.kv("ns-collections", scope="hub", writable=True, declare=True),)\n'
        )
        (tmp_path / "sanctioned_declare.py").write_text(source, encoding="utf-8")

        assert find_declare_outside_the_declarer((tmp_path,), _SANCTIONED_DECLARERS) == []

    def test_scope_walker_flags_an_unscoped_shared_bucket(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.subject_permissions import JsResource\n\n\n"
            "def _new_principal(ns):\n"
            '    return (JsResource.kv(f"{ns}-collections", scope=None, writable=True),)\n'
        )
        (tmp_path / "planted_unscoped.py").write_text(source, encoding="utf-8")

        violations = find_unscoped_shared_bucket_grants((tmp_path,))

        assert [v.symbol for v in violations] == ["_new_principal"]

    def test_scope_walker_accepts_a_scoped_shared_bucket_and_an_unscoped_neighbour(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.subject_permissions import JsResource\n\n\n"
            "def _new_principal(ns, scope):\n"
            "    return (\n"
            '        JsResource.kv(f"{ns}-collections", scope=scope, writable=True),\n'
            '        JsResource.kv("checkpoints", scope=None, writable=True),\n'
            "    )\n"
        )
        (tmp_path / "sanctioned_scoped.py").write_text(source, encoding="utf-8")

        assert find_unscoped_shared_bucket_grants((tmp_path,)) == []

    def test_decision_walker_flags_an_omitted_keyword(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.subject_permissions import JsResource\n\n\n"
            "def _new_principal(ns):\n"
            '    return (JsResource.kv(f"{ns}-epochs", scope=None),)\n'
        )
        (tmp_path / "planted_implicit.py").write_text(source, encoding="utf-8")

        violations = find_implicit_bucket_decisions((tmp_path,))

        assert [v.symbol for v in violations] == ["_new_principal"]
        assert "writable" in violations[0].reason

    def test_decision_walker_accepts_a_complete_declaration(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.subject_permissions import JsResource\n\n\n"
            "def _new_principal(ns):\n"
            '    return (JsResource.kv(f"{ns}-epochs", scope=None, writable=True),)\n'
        )
        (tmp_path / "sanctioned_complete.py").write_text(source, encoding="utf-8")

        assert find_implicit_bucket_decisions((tmp_path,)) == []
