"""enforcement walker: every L2-live collection registry is given a key scope (coll-task-06a).

``coll-task-03`` made ``{scope}.{table}.{body}`` the shape of every L2 key and made
``CollectionRegistry.configure`` refuse an L2 client with no ``kv_key_scope``. That refusal
covers the registry-DEFAULT wiring path and nothing else: a registry configured with
``l1_backend`` alone, whose collections each receive their own ``nats_client=``, is L2-live and
never passes through the check -- the explicit client on a collection WINS over the registry
default. It fails later, in :meth:`BaseCollection.l2_key`'s backstop, on the first cache access
under load.

This gate closes that gap statically. It walks every ``CollectionRegistry()`` construction under
this repo's ``packages/*/src`` trees, decides whether the registry is L2-live, and requires a
``configure(kv_key_scope=...)`` naming it in the same module.

**Five wiring shapes, not four.** A narrower reader reproduces the exact blind spot that hid four
processes from an early sweep of this work:

1. ``registry.configure(l2_client=nc)`` -- the registry default;
2. ``registry.register(..., l2_client=nc)``;
3. ``registry.bind_table(..., l2_client=nc)``;
4. ``Coll(registry=registry, config=cfg, nats_client=nc)`` -- the per-collection keyword, which
   wins over the registry default and is how ``registry/rbac_stack.py`` runs five L2-live ACL
   collections on a registry ``configure()`` never saw a client for;
5. ``FeatureCache(registry, config, nats_client, None)`` -- the same client passed POSITIONALLY,
   which a keyword-only reader does not see at all.

**Scope is the MODULE, deliberately**, mirroring the hub's sibling gate: liveness evidence and the
``configure`` call routinely sit in different functions of one file. The cost is that two
same-named registries in one module read as one, which errs toward demanding MORE, never less --
and the per-REGISTRY keying still separates two DIFFERENTLY-named registries in one module, which
is exactly the registry server's shape.

Mode via ``L2_SCOPE_WIRING_ENFORCEMENT_MODE`` (default ``strict``). Exemptions live in
``_l2_scope_wiring_exemptions.txt`` and require a specific ``# rationale:`` line.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Final

import pytest

from threetears.enforcement.common import (
    CLIENT_KEYWORDS,
    MODE_REPORT,
    MODE_STRICT,
    Violation,
    apply_exemptions,
    argument_spellings,
    constructed_registry_lines,
    emit_report,
    find_local_src_roots,
    iter_python_files,
    l2_live_registries,
    parse_exemptions_with_rationale,
    parse_python_file,
    receiver,
    resolve_mode,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[4]
_EXEMPTIONS_PATH: Final[Path] = Path(__file__).resolve().parent / "_l2_scope_wiring_exemptions.txt"
_MODE_ENV_VAR: Final[str] = "L2_SCOPE_WIRING_ENFORCEMENT_MODE"

_CATEGORY: Final[str] = "l2_scope.live_registry_without_a_scope"

#: the registry class, the binder methods, the client keywords and the positional
#: client spellings all live in ``enforcement.common.collection_registry`` now, shared
#: with the invalidation-listener gate. They were duplicated here, and a local copy of
#: ``CLIENT_SPELLINGS`` in particular goes stale silently: it is a heuristic list that
#: grows, and ruff does not flag an unused module constant, so the copy keeps compiling
#: while the gate it feeds stops seeing a wiring form the other gate learned.

#: the keyword that supplies the scope.
_SCOPE_KEYWORD: Final[str] = "kv_key_scope"

#: floor on the real scope. a reader that stopped matching would demand nothing of anybody while
#: still reporting green, so the live count is asserted from below as well as from above.
_MINIMUM_LIVE_REGISTRIES: Final[int] = 2


def scoped_registries(tree: ast.AST) -> frozenset[str]:
    """return every registry a ``configure(kv_key_scope=...)`` call names.

    An explicit ``kv_key_scope=None`` is the "leave unchanged" argument and supplies nothing, so
    it does not count -- writing it would otherwise silence this gate while changing no state.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: spellings named by a scope-supplying call
    :rtype: frozenset[str]
    """
    scoped: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        supplies = False
        for keyword in node.keywords:
            if keyword.arg != _SCOPE_KEYWORD:
                continue
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
                continue
            supplies = True
        if not supplies:
            continue
        call_receiver = receiver(node)
        if call_receiver is not None:
            scoped.add(call_receiver)
        scoped.update(argument_spellings(node))
    return frozenset(scoped)


def unscoped_live_registries(tree: ast.AST) -> list[str]:
    """return every L2-live registry in a module that no scope call names.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: offending registry spellings, sorted
    :rtype: list[str]
    """
    return sorted(l2_live_registries(tree) - scoped_registries(tree))


_ORDER_CATEGORY: Final[str] = "l2_scope.client_configured_before_scope"


def client_before_scope_registries(tree: ast.AST) -> dict[str, int]:
    """return every registry given an L2 client by a call that precedes its scope.

    ``configure()`` MERGES, and evaluates its refusal over the merged state at the end of
    each call -- so two-pass wiring works in one order only. Scope first is fine: that call
    ends with a scope and no client, the next ends with both. Client first RAISES, because
    that call ends with a client and no scope and cannot know a later call intends to
    supply one.

    :func:`unscoped_live_registries` cannot see this: it is a set difference over the whole
    module, so a registry that is scoped ANYWHERE counts as scoped. A module wired
    client-first passes that gate green and dies at startup, which is the failure this gate
    exists to move earlier.

    **Ordering is by ``lineno`` across the whole module, ignoring function boundaries.**
    A registry whose scope call lives in a helper DEFINED below its client-wiring call
    site would read as client-first and be a false positive. Every ``configure(`` site in
    this repo today supplies the scope in the same call, so nothing hits it; the day one
    does, narrow the comparison to a shared enclosing scope rather than exempting the
    file -- an exemption here silences the real check too.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: registry spelling -> line of the offending client-first call
    :rtype: dict[str, int]
    """
    first_scope: dict[str, int] = {}
    client_only: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_receiver = receiver(node)
        if call_receiver is None:
            continue
        supplies_scope = any(
            kw.arg == _SCOPE_KEYWORD and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
            for kw in node.keywords
        )
        supplies_client = any(
            kw.arg in CLIENT_KEYWORDS and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
            for kw in node.keywords
        )
        if supplies_scope:
            prior = first_scope.get(call_receiver)
            if prior is None or node.lineno < prior:
                first_scope[call_receiver] = node.lineno
        elif supplies_client:
            client_only.setdefault(call_receiver, []).append(node.lineno)

    offenders: dict[str, int] = {}
    for offender, lines in client_only.items():
        scope_line = first_scope.get(offender)
        # no scope call at all is `unscoped_live_registries`' finding, not this one --
        # reporting both would double-count one defect under two categories.
        if scope_line is None:
            continue
        earlier = [line for line in lines if line < scope_line]
        if earlier:
            offenders[offender] = min(earlier)
    return offenders


def find_client_before_scope(scan_roots: tuple[Path, ...]) -> list[Violation]:
    """flag every registry handed an L2 client before it is given a scope.

    :param scan_roots: src roots to scan
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
            violations.extend(
                Violation(
                    category=_ORDER_CATEGORY,
                    file=source,
                    line=line,
                    symbol=name,
                    reason=(
                        f"'{name}' is given an {'/'.join(sorted(CLIENT_KEYWORDS))} here, before any "
                        f"call supplies its {_SCOPE_KEYWORD}. configure() evaluates its refusal over "
                        f"the merged state at the END of each call, so this one raises "
                        f"L2ScopeNotConfiguredError at startup -- it cannot know a later call intends "
                        f"to supply a scope. order the scope first, or pass both together"
                    ),
                )
                for name, line in sorted(client_before_scope_registries(tree).items())
            )
    return violations


def find_unscoped_registries(scan_roots: tuple[Path, ...]) -> list[Violation]:
    """flag every L2-live registry that is never given a ``kv_key_scope``.

    :param scan_roots: src roots to scan
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
            offenders = unscoped_live_registries(tree)
            if not offenders:
                continue
            lines = constructed_registry_lines(tree)
            violations.extend(
                Violation(
                    category=_CATEGORY,
                    file=source,
                    line=lines.get(name, 1),
                    symbol=name,
                    reason=(
                        f"'{name}' holds an L2-live collection and is never given a "
                        f"{_SCOPE_KEYWORD}; every key it writes lands on the shared collections "
                        "bucket under no principal segment, which no minted grant can name. pass "
                        f"{_SCOPE_KEYWORD}=threetears.nats.kv_key_scope_for(...) at the "
                        "configure() call"
                    ),
                )
                for name in offenders
            )
    return violations


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


class TestEveryLiveRegistryCarriesAScope:
    """the gate itself, over every registry construction in this repo's src trees."""

    def test_every_l2_live_registry_is_scoped(self) -> None:
        """an unscoped live registry writes keys no grant can name."""
        _assert_clean(find_unscoped_registries(find_local_src_roots(_REPO_ROOT)), _CATEGORY)

    def test_no_registry_is_given_a_client_before_its_scope(self) -> None:
        """client-first two-pass wiring raises at startup; being scoped later is too late."""
        _assert_clean(find_client_before_scope(find_local_src_roots(_REPO_ROOT)), _ORDER_CATEGORY)


class TestTheReaderIsNotVacuous:
    """a reader that matched nothing would report green over the whole gap."""

    def test_the_repo_really_does_run_live_registries(self) -> None:
        """the count is asserted from below so a broken reader is loud."""
        live = sum(
            len(l2_live_registries(tree))
            for root in find_local_src_roots(_REPO_ROOT)
            for source in iter_python_files(root)
            if (tree := parse_python_file(source)) is not None
        )
        assert live >= _MINIMUM_LIVE_REGISTRIES, (
            f"only {live} L2-live registries found across packages/*/src; the wiring reader has "
            "probably stopped matching one of the five shapes"
        )

    def test_both_registries_in_the_registry_server_process_are_seen(self) -> None:
        """the two-registries-in-one-process shape, asserted rather than assumed.

        ``registry/server.py`` configures ``l2_client=`` directly; ``rbac_stack.py`` configures
        ``l1_backend`` + ``l3_pool`` ONLY and then builds five ACL collections each with its own
        ``nats_client=``. An ``l2_client=``-only sweep sees the first and misses the second.
        """
        package = _REPO_ROOT / "packages" / "registry" / "src" / "threetears" / "registry"
        for module in ("server.py", "rbac_stack.py"):
            tree = parse_python_file(package / module)
            assert tree is not None
            assert l2_live_registries(tree), f"{module} no longer reads as holding an L2-live registry"

    def test_an_l3_only_registry_is_not_dragged_in(self) -> None:
        """a registry with no live client must not be demanded a scope."""
        tree = parse_python_file(
            _REPO_ROOT / "packages" / "agent" / "wake" / "src" / "threetears" / "agent" / "wake" / "tick.py"
        )
        assert tree is not None
        assert not l2_live_registries(tree), "wake/tick.py reads as L2-live but wires no client"


class TestTheReaderCatchesEachWiringShape:
    """the fallibility proofs: one planted violation per shape the reader must report."""

    def test_shape_one_configure_l2_client(self) -> None:
        """the registry-default binder."""
        tree = ast.parse("registry = CollectionRegistry()\nregistry.configure(l3_pool=pool, l2_client=nc)\n")
        assert unscoped_live_registries(tree) == ["registry"]

    def test_shape_two_register_l2_client(self) -> None:
        """the per-table binder."""
        tree = ast.parse("registry = CollectionRegistry()\nregistry.register('widgets', l2_client=nc)\n")
        assert unscoped_live_registries(tree) == ["registry"]

    def test_shape_three_bind_table_l2_client(self) -> None:
        """the bind_table binder."""
        tree = ast.parse("registry = CollectionRegistry()\nregistry.bind_table('widgets', l2_client=nc)\n")
        assert unscoped_live_registries(tree) == ["registry"]

    def test_shape_four_collection_nats_client_keyword(self) -> None:
        """the rbac-stack shape: L1/L3-only registry, live client on each collection."""
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l1_backend=backend, l3_pool=pool)\n"
            "groups = GroupCollection(registry=registry, config=cfg, nats_client=nats_client)\n"
        )
        assert unscoped_live_registries(tree) == ["registry"]

    def test_shape_five_positional_nats_client(self) -> None:
        """the geo shape: registry and client both passed positionally."""
        tree = ast.parse("registry = CollectionRegistry()\ncache = FeatureCache(registry, config, nats_client, None)\n")
        assert unscoped_live_registries(tree) == ["registry"]

    def test_a_second_registry_in_one_module_is_caught(self) -> None:
        """per-REGISTRY keying: one scope must not cover the other."""
        tree = ast.parse(
            "first = CollectionRegistry()\n"
            "first.configure(l2_client=nc, kv_key_scope=scope)\n"
            "second = CollectionRegistry()\n"
            "second.configure(l2_client=nc)\n"
        )
        assert unscoped_live_registries(tree) == ["second"]

    def test_an_attribute_bound_registry_is_caught(self) -> None:
        """a registry stored on ``self`` is a registry all the same."""
        tree = ast.parse("self._registry = CollectionRegistry()\nself._registry.configure(l2_client=self._nc)\n")
        assert unscoped_live_registries(tree) == ["self._registry"]

    def test_an_explicit_none_scope_does_not_count(self) -> None:
        """``kv_key_scope=None`` is the leave-unchanged argument and supplies nothing."""
        tree = ast.parse("registry = CollectionRegistry()\nregistry.configure(l2_client=nc, kv_key_scope=None)\n")
        assert unscoped_live_registries(tree) == ["registry"]


class TestTheReaderLeavesSanctionedWiringAlone:
    """a walker that flags everything is as useless as one that flags nothing."""

    def test_a_scoped_registry_is_left_alone(self) -> None:
        """the shape every production site must land on."""
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l1_backend=backend, l2_client=nc, "
            "kv_key_scope=kv_key_scope_for(Principal.REGISTRY))\n"
        )
        assert unscoped_live_registries(tree) == []

    def test_a_two_pass_wiring_is_left_alone(self) -> None:
        """scope first, client later -- ``configure()`` merges, so this is a normal shape."""
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(kv_key_scope=scope)\n"
            "registry.configure(l2_client=nc)\n"
        )
        assert unscoped_live_registries(tree) == []
        assert client_before_scope_registries(tree) == {}

    def test_an_l1_only_registry_is_left_alone(self) -> None:
        """a process with no L2 tier at all is a valid configuration."""
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l1_backend=backend)\n"
            "coll = PodAffinityCollection(registry=registry, config=cfg, nats_client=None)\n"
        )
        assert unscoped_live_registries(tree) == []

    def test_an_l3_only_registry_is_left_alone(self) -> None:
        """the wake / scrape shape: an L3 pool and nothing else."""
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l3_pool=pool)\n"
            "fires = WakeFireCollection(registry=registry, config=cfg)\n"
        )
        assert unscoped_live_registries(tree) == []


class TestTwoPassWiringHasOnlyOneWorkingOrder:
    """``configure()`` refuses over the MERGED state at the end of each call."""

    def test_client_before_scope_is_flagged(self) -> None:
        """that first call ends with a client and no scope, so it raises at startup."""
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l2_client=nc)\n"
            "registry.configure(kv_key_scope=scope)\n"
        )

        assert client_before_scope_registries(tree) == {"registry": 2}

    def test_the_presence_gate_cannot_see_it(self) -> None:
        """the reason this walker exists: a set difference has no order.

        Without this assertion the new walker looks redundant, and the next reader
        deletes it.
        """
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l2_client=nc)\n"
            "registry.configure(kv_key_scope=scope)\n"
        )

        assert unscoped_live_registries(tree) == []

    def test_both_in_one_call_is_left_alone(self) -> None:
        """the single-call shape ends with both, so it never sees the refusal."""
        tree = ast.parse("registry = CollectionRegistry()\nregistry.configure(l2_client=nc, kv_key_scope=scope)\n")

        assert client_before_scope_registries(tree) == {}

    def test_a_registry_with_no_scope_at_all_is_the_other_gate(self) -> None:
        """one defect, one category -- reporting both would double-count it."""
        tree = ast.parse("registry = CollectionRegistry()\nregistry.configure(l2_client=nc)\n")

        assert client_before_scope_registries(tree) == {}
        assert unscoped_live_registries(tree) == ["registry"]

    def test_an_explicit_none_scope_does_not_count_as_supplying_one(self) -> None:
        """``kv_key_scope=None`` is the leave-unchanged argument and changes no state."""
        tree = ast.parse(
            "registry = CollectionRegistry()\nregistry.configure(l2_client=nc)\nregistry.configure(kv_key_scope=None)\n"
        )

        assert client_before_scope_registries(tree) == {}


class TestTheExemptionFileIsDisciplined:
    """an exemption file nobody has to justify is an off switch."""

    def test_the_committed_exemptions_parse(self) -> None:
        """the real file clears the rationale discipline."""
        parse_exemptions_with_rationale(_EXEMPTIONS_PATH)
