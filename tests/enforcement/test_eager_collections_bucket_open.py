"""enforcement walker: an L2-live process opens the shared bucket EAGERLY (``coll-task-04a`` KVC-05).

``coll-task-06a`` wired the eager bind into the registry server; ``coll-task-07c`` wires it into the
tool-pod bootstrap. Nothing statically requires the NEXT process to do it, and the omission is
silent in both directions, which is what makes it enforceable rather than reviewable:

- **without the eager open, a config mismatch lands in a request path.**
  ``BaseCollection._ensure_kv`` resolves the bucket on the FIRST READ, so a bucket carrying a
  configuration the process refuses (``allow_direct`` unset -- which puts every KV read back on the
  body-carried ``$JS.API.STREAM.MSG.GET`` form that no key-scoped ``$KV.`` grant can constrain)
  surfaces as a ``KvConfigMismatch`` raised under load rather than at startup. That is exactly the
  failure the distinct exception type was introduced to avoid, relocated rather than fixed.
- **without it, the process's own opens can disagree.** ``NatsClient.kv_bucket`` caches by full
  bucket name, so whichever call site opens first decides the configuration for the life of the
  process -- and a process with two registries (the registry server has two, and the hub has more)
  then has no defined answer to "which config is live".

**The rule.** A module that binds the registry-DEFAULT L2 client -- ``configure(l2_client=...)`` --
must also name :func:`threetears.core.collections.bucket.bind_collections_bucket` or
``ensure_kv_bucket`` somewhere in the same module. Module scope, deliberately, mirroring the sibling
L2-scope gate: the eager open and the ``configure`` call routinely sit in different functions of one
file, and demanding they share a function would reject the registry server's own correct shape.

**What it does NOT cover, stated because a green gate invites over-trust.** A registry that is
L2-live only through per-collection ``nats_client=`` arguments (``registry/rbac_stack.py``, the
hub's ``gateway/acl.py``) never calls ``configure(l2_client=)`` and is not matched here. That is not
an oversight: those registries share a PROCESS with one that does, and the bucket handle is
process-wide, so a second eager open in a second module would be redundant rather than safer. The
gate is per-module and the property is per-process; the sibling
``packages/nats/tests/enforcement/test_l2_scope_wiring.py`` is the walker that covers every wiring
shape, and it exists because a narrower reader hid four processes from an early sweep of this work.

Mode via ``EAGER_BUCKET_OPEN_ENFORCEMENT_MODE`` (default ``strict``). Exemptions live in
``_eager_bucket_open_exemptions.txt`` and require a specific ``# rationale:`` line.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Final

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

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_EXEMPTIONS_PATH: Final[Path] = _REPO_ROOT / "tests" / "enforcement" / "_eager_bucket_open_exemptions.txt"
_MODE_ENV_VAR: Final[str] = "EAGER_BUCKET_OPEN_ENFORCEMENT_MODE"

_CATEGORY: Final[str] = "eager_bucket_open.l2_wiring_without_an_eager_bind"


#: the keyword whose presence makes a ``configure`` call the registry-DEFAULT L2 wiring.
_L2_CLIENT_KEYWORD: Final[str] = "l2_client"

#: the method that binds it.
_CONFIGURE: Final[str] = "configure"

#: either of these names, called or merely referenced, satisfies the rule. Two rather than one
#: because a process may reach the primitive directly (``ensure_kv_bucket``) or through the shared
#: bounded-retry helper -- and a module that delegates to a helper it imported still NAMES it.
_EAGER_OPENERS: Final[frozenset[str]] = frozenset({"bind_collections_bucket", "ensure_kv_bucket"})


def _called_name(func: ast.expr) -> str | None:
    """the bare callee name of a call target, however it was imported or attributed.

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


def _l2_configure_calls(tree: ast.Module) -> list[ast.Call]:
    """every ``*.configure(..., l2_client=...)`` call in one module.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :return: the matching call nodes, in source order
    :rtype: list[ast.Call]
    """
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _called_name(node.func) != _CONFIGURE:
            continue
        if any(keyword.arg == _L2_CLIENT_KEYWORD for keyword in node.keywords):
            found.append(node)
    return found


def _names_an_eager_opener(tree: ast.Module) -> bool:
    """whether one module mentions an eager bucket-open primitive at all.

    Deliberately a NAME check rather than a call-ordering check. Ordering is a property of one
    process's startup path, which routinely spans several functions and a caller in another module;
    a walker asserting it would either reject correct code or encode one process's shape as the
    rule. What is checkable, and what the omission actually looks like, is a module that wires L2
    and never mentions the open at all.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :return: whether any eager-open name appears
    :rtype: bool
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _EAGER_OPENERS:
            return True
        if isinstance(node, ast.Attribute) and node.attr in _EAGER_OPENERS:
            return True
        if isinstance(node, ast.alias) and node.name.split(".")[-1] in _EAGER_OPENERS:
            return True
    return False


def find_l2_wiring_without_an_eager_bind(scan_roots: tuple[Path, ...]) -> list[Violation]:
    """flag every module that binds the registry-default L2 client and never opens the bucket.

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
            calls = _l2_configure_calls(tree)
            if not calls or _names_an_eager_opener(tree):
                continue
            violations.append(
                Violation(
                    category=_CATEGORY,
                    file=source,
                    line=calls[0].lineno,
                    symbol=source.stem,
                    reason=(
                        "this module wires an L2 client onto a CollectionRegistry and never opens "
                        "the shared collections bucket. the bucket then resolves lazily, on the "
                        "first cache read, so a configuration this process refuses raises inside a "
                        "request path under load instead of at startup -- and whichever call site "
                        "opens first silently decides the config for the whole process, because "
                        "NatsClient.kv_bucket caches by name. call bind_collections_bucket(nc) "
                        "immediately after connect and before configure()"
                    ),
                )
            )
    return violations


def _assert_clean(violations: list[Violation]) -> None:
    """apply exemptions + mode and fail in strict mode with a rendered report.

    :param violations: raw walker output
    :ptype violations: list[Violation]
    :return: nothing
    :rtype: None
    :raises pytest.fail.Exception: in strict mode with surviving violations
    """
    exemptions = parse_exemptions_with_rationale(_EXEMPTIONS_PATH)
    filtered = apply_exemptions(violations, exemptions, _REPO_ROOT)
    mode = resolve_mode(_MODE_ENV_VAR, default=MODE_STRICT)
    report = emit_report(filtered, (_REPO_ROOT,), exemptions, mode, _REPO_ROOT, domain=_CATEGORY)
    print(report, file=sys.stderr)
    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"{_CATEGORY} found {len(filtered)} violation(s):\n{report}")


class TestEagerCollectionsBucketOpen:
    """the rule over this repo's own src trees."""

    def test_every_l2_wiring_module_opens_the_bucket_eagerly(self) -> None:
        _assert_clean(find_l2_wiring_without_an_eager_bind(find_local_src_roots(_REPO_ROOT)))

    def test_the_rule_is_not_vacuous(self) -> None:
        """the two real modules under this rule are both found, and both satisfy it.

        A gate whose walker matches nothing reads exactly like a gate with nothing to report, and
        would keep passing if the last L2 wiring site were deleted or renamed out from under it.
        The tool-pod bootstrap is named EXPLICITLY because it lives in the nested
        ``packages/agent/*`` tree, which the shared root helper walked past until the helper was
        widened to recurse to any depth. Without this assertion the gate would report clean over
        the one module it was written for.
        """
        matched = [
            source.relative_to(_REPO_ROOT).as_posix()
            for root in find_local_src_roots(_REPO_ROOT)
            for source in iter_python_files(root)
            if (tree := parse_python_file(source)) is not None and _l2_configure_calls(tree)
        ]
        assert "packages/registry/src/threetears/registry/server.py" in matched, matched
        assert "packages/agent/tools/src/threetears/agent/tools/bootstrap.py" in matched, matched


class TestTheWalkerFlagsAPlantedViolation:
    """the walker must fire on a planted violation and stay quiet on the sanctioned shape."""

    def test_it_flags_l2_wiring_with_no_eager_open(self, tmp_path: Path) -> None:
        source = (
            "from threetears.core.collections.registry import CollectionRegistry\n\n\n"
            "def build_stack(nc, scope):\n"
            "    registry = CollectionRegistry()\n"
            "    registry.configure(l2_client=nc, kv_key_scope=scope)\n"
            "    return registry\n"
        )
        (tmp_path / "planted_lazy.py").write_text(source, encoding="utf-8")

        violations = find_l2_wiring_without_an_eager_bind((tmp_path,))

        assert [v.symbol for v in violations] == ["planted_lazy"]
        assert "lazily" in violations[0].reason

    def test_it_accepts_a_module_that_binds_through_the_shared_helper(self, tmp_path: Path) -> None:
        source = (
            "from threetears.core.collections import bind_collections_bucket\n"
            "from threetears.core.collections.registry import CollectionRegistry\n\n\n"
            "async def build_stack(nc, scope):\n"
            "    await bind_collections_bucket(nc)\n"
            "    registry = CollectionRegistry()\n"
            "    registry.configure(l2_client=nc, kv_key_scope=scope)\n"
            "    return registry\n"
        )
        (tmp_path / "sanctioned_helper.py").write_text(source, encoding="utf-8")

        assert find_l2_wiring_without_an_eager_bind((tmp_path,)) == []

    def test_it_accepts_a_module_that_calls_the_primitive_directly(self, tmp_path: Path) -> None:
        source = (
            "from threetears.core.collections.registry import CollectionRegistry\n\n\n"
            "async def build_stack(nc, scope):\n"
            '    await nc.ensure_kv_bucket(name="collections", create_if_missing=False)\n'
            "    registry = CollectionRegistry()\n"
            "    registry.configure(l2_client=nc, kv_key_scope=scope)\n"
            "    return registry\n"
        )
        (tmp_path / "sanctioned_primitive.py").write_text(source, encoding="utf-8")

        assert find_l2_wiring_without_an_eager_bind((tmp_path,)) == []

    def test_it_ignores_a_registry_that_wires_no_l2_client(self, tmp_path: Path) -> None:
        """an L1-only or L3-only registry opens no bucket and must not be dragged into this rule."""
        source = (
            "from threetears.core.collections.registry import CollectionRegistry\n\n\n"
            "def build_stack(l1, pool):\n"
            "    registry = CollectionRegistry()\n"
            "    registry.configure(l1_backend=l1, l3_pool=pool)\n"
            "    return registry\n"
        )
        (tmp_path / "l1_only.py").write_text(source, encoding="utf-8")

        assert find_l2_wiring_without_an_eager_bind((tmp_path,)) == []
