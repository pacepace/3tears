"""no-bespoke-reuse enforcement guard (gu-task-06, GU-06-07 — 3tears half).

the Grand Unification epic has exactly ONE new component (the OpenAPI parser,
gu-task-23); everything else reuses an existing primitive. this guard is the durable
encoding of the epic reuse-map: it fails if any class under a 3tears package hand-rolls
one of the three patterns the platform already solves:

(a) a buffer + periodic-flush loop — reuse ``ConversationWriteBuffer``
    (``threetears.conversations.buffer``);
(b) a raw, long-lived ``httpx.AsyncClient`` / ``httpx.Client`` held as a field outside
    the sanctioned traced HTTP-client wrapper (``threetears.core.http_client``);
(c) a KV / counter that stores an ``SQLiteBackend`` and exposes cache-style verbs
    without subclassing ``BaseCollection`` — reuse the Collection primitive.

check (c) DELEGATES to the shared, already-proven ``find_wrapper_classes`` walker in
``threetears.enforcement.cache`` (reuse, not a re-rolled walker — the same rule this
guard enforces). checks (a) and (b) are small AST walkers local to this module, scoped
to STORED fields (a long-lived bespoke buffer / client), not one-shot ephemeral usage.

mode via ``REUSE_ENFORCEMENT_MODE`` (default ``strict``), mirroring
``CACHE_ENFORCEMENT_MODE``. exemptions live in ``_no_bespoke_reuse_exemptions.txt`` and
require a specific ``# rationale:`` line. the sanctioned reuse targets are allowlisted in
the walkers themselves — they are the components bespoke code should reuse.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from threetears.enforcement.cache import find_wrapper_classes
from threetears.enforcement.common import (
    MODE_REPORT,
    MODE_STRICT,
    Violation,
    apply_exemptions,
    discover_src_roots,
    emit_report,
    find_local_src_roots,
    iter_python_files,
    parse_exemptions_with_rationale,
    parse_python_file,
    relative_posix_path,
    resolve_mode,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXEMPTIONS_PATH = _REPO_ROOT / "tests" / "enforcement" / "_no_bespoke_reuse_exemptions.txt"
_MODE_ENV_VAR = "REUSE_ENFORCEMENT_MODE"

_FLUSH_LOOP_CATEGORY = "reuse.bespoke_flush_loop"
_HTTPX_CLIENT_CATEGORY = "reuse.bespoke_httpx_client"

# the sanctioned reuse targets — the components bespoke code is meant to reuse, not
# exceptions to the rule. paths are repo-relative posix.
_SANCTIONED_FLUSH_SITES: frozenset[str] = frozenset(
    {
        # the platform's periodic write-behind buffer primitive. bespoke buffer+flush
        # loops must reuse THIS rather than re-rolling a timer loop over an accumulator.
        "packages/conversations/src/threetears/conversations/buffer.py",
    }
)
_SANCTIONED_HTTPX_SITES: frozenset[str] = frozenset(
    {
        # the sanctioned traced HTTP-client wrapper — the ONE place a long-lived
        # httpx.AsyncClient is owned + wrapped with tracing/retry.
        "packages/core/src/threetears/core/http_client.py",
        # the mcp package's single-purpose HTTP transport wrapper (its own traced
        # client module); the anti-pattern is a SERVICE holding a bespoke client, not
        # a dedicated transport module.
        "packages/mcp/src/threetears/mcp/http_client.py",
    }
)

# check-(c) reuse knobs, mirroring the cache-enforcement defaults exactly so the two
# guards agree on what counts as a cache-verb wrapper (the proven-green set deliberately
# excludes ``remove`` — the sanctioned ``WriteBuffer`` flush primitive exposes it).
_BASE_COLLECTION_NAMES: frozenset[str] = frozenset({"BaseCollection"})
_CACHE_METHOD_NAMES: frozenset[str] = frozenset({"get", "put", "set", "delete", "upsert"})


# ---------------------------------------------------------------------------
# check (a) — bespoke buffer + periodic-flush loop
# ---------------------------------------------------------------------------


def _is_sleep_call(func: ast.expr) -> bool:
    """true iff ``func`` names a ``*.sleep`` call (``asyncio.sleep`` / ``time.sleep``).

    :param func: the ``ast.Call.func`` node under inspection
    :ptype func: ast.expr
    :return: whether the call target is a ``.sleep`` attribute access
    :rtype: bool
    """
    return isinstance(func, ast.Attribute) and func.attr == "sleep"


def _class_has_periodic_loop(node: ast.ClassDef) -> bool:
    """true iff the class contains a ``while`` loop whose body calls ``*.sleep``.

    :param node: class definition to inspect
    :ptype node: ast.ClassDef
    :return: whether a periodic (sleep-driven) loop is present
    :rtype: bool
    """
    result = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.While):
            for inner in ast.walk(sub):
                if isinstance(inner, ast.Call) and _is_sleep_call(inner.func):
                    result = True
                    break
        if result:
            break
    return result


def _self_attr(value: ast.expr) -> str | None:
    """return the attribute name if ``value`` is ``self.<attr>``, else ``None``.

    :param value: expression to test
    :ptype value: ast.expr
    :return: the attribute name, or ``None`` when ``value`` is not ``self.<attr>``
    :rtype: str | None
    """
    result: str | None = None
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self":
        result = value.attr
    return result


def _class_accumulation_buffer(node: ast.ClassDef) -> str | None:
    """return the name of a ``self`` buffer the class accumulates into, else ``None``.

    matches ``self.<buf>.append(...)`` / ``.appendleft(...)`` and subscript stores
    ``self.<buf>[key] = ...`` — the two long-lived-accumulator shapes.

    :param node: class definition to inspect
    :ptype node: ast.ClassDef
    :return: the buffer attribute name, or ``None`` when the class does not accumulate
    :rtype: str | None
    """
    result: str | None = None
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"append", "appendleft"}
        ):
            buffer_name = _self_attr(sub.func.value)
            if buffer_name is not None:
                result = buffer_name
                break
        if isinstance(sub, ast.Assign):
            hit = False
            for target in sub.targets:
                if isinstance(target, ast.Subscript):
                    buffer_name = _self_attr(target.value)
                    if buffer_name is not None:
                        result = buffer_name
                        hit = True
                        break
            if hit:
                break
    return result


def find_bespoke_flush_loops(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    allowed_sites: frozenset[str],
) -> list[Violation]:
    """flag classes that hand-roll a buffer + periodic-flush loop.

    fingerprint: the class BOTH accumulates into a ``self`` buffer AND drives a
    ``while`` loop with a ``*.sleep`` tick. that pair is the write-behind buffer the
    platform already ships as ``ConversationWriteBuffer``; bespoke re-rolls must reuse
    it. sanctioned reuse targets in ``allowed_sites`` are skipped.

    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root for relative-path rendering
    :ptype repo_root: Path
    :param allowed_sites: repo-relative posix paths of sanctioned primitives to skip
    :ptype allowed_sites: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            if relative_posix_path(source, repo_root) in allowed_sites:
                continue
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                buffer_name = _class_accumulation_buffer(node)
                if buffer_name is None:
                    continue
                if not _class_has_periodic_loop(node):
                    continue
                violations.append(
                    Violation(
                        category=_FLUSH_LOOP_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=node.name,
                        reason=(
                            f"class '{node.name}' hand-rolls a buffer ('self.{buffer_name}') "
                            "plus a periodic sleep loop; reuse "
                            "threetears.conversations.buffer.ConversationWriteBuffer instead"
                        ),
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# check (b) — bespoke long-lived httpx client held as a field
# ---------------------------------------------------------------------------


def _names_httpx_client(value: ast.expr) -> bool:
    """true iff ``value`` names ``httpx.AsyncClient`` / ``httpx.Client`` (or bare).

    :param value: expression to test (a call target or an annotation)
    :ptype value: ast.expr
    :return: whether the expression references an httpx client type
    :rtype: bool
    """
    result = False
    if isinstance(value, ast.Attribute) and value.attr in {"AsyncClient", "Client"}:
        result = isinstance(value.value, ast.Name) and value.value.id == "httpx"
    elif isinstance(value, ast.Name) and value.id in {"AsyncClient", "Client"}:
        result = True
    return result


def _class_stores_httpx_client(node: ast.ClassDef) -> bool:
    """true iff the class stores an httpx client on ``self`` (constructed or annotated).

    matches ``self.<x> = httpx.AsyncClient(...)`` and annotated self-stores
    ``self.<x>: httpx.AsyncClient = ...`` — a long-lived client held as a field. one-shot
    ephemeral ``async with httpx.AsyncClient() as c`` usage is intentionally NOT matched.

    :param node: class definition to inspect
    :ptype node: ast.ClassDef
    :return: whether the class holds an httpx client field
    :rtype: bool
    """
    result = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            stores_self = any(_self_attr(target) is not None for target in sub.targets)
            if stores_self and isinstance(sub.value, ast.Call) and _names_httpx_client(sub.value.func):
                result = True
                break
        if isinstance(sub, ast.AnnAssign):
            if _self_attr(sub.target) is not None and _names_httpx_client(sub.annotation):
                result = True
                break
    return result


def find_bespoke_httpx_clients(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    allowed_sites: frozenset[str],
) -> list[Violation]:
    """flag classes that hold a raw, long-lived httpx client field.

    the sanctioned traced HTTP-client wrapper modules in ``allowed_sites`` are skipped;
    every other class holding an ``httpx.AsyncClient`` / ``httpx.Client`` field is a
    bespoke client that should route through the traced wrapper.

    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root for relative-path rendering
    :ptype repo_root: Path
    :param allowed_sites: repo-relative posix paths of sanctioned wrapper modules to skip
    :ptype allowed_sites: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            if relative_posix_path(source, repo_root) in allowed_sites:
                continue
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if not _class_stores_httpx_client(node):
                    continue
                violations.append(
                    Violation(
                        category=_HTTPX_CLIENT_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=node.name,
                        reason=(
                            f"class '{node.name}' holds a raw httpx client field; route through "
                            "the traced wrapper (threetears.core.http_client) instead"
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


class TestNoBespokeReuse:
    """the three reuse-map checks over the 3tears package tree."""

    def test_no_bespoke_flush_loops(self) -> None:
        """no class hand-rolls a buffer + periodic-flush loop."""
        scan_roots = find_local_src_roots(_REPO_ROOT)
        violations = find_bespoke_flush_loops(scan_roots, _REPO_ROOT, _SANCTIONED_FLUSH_SITES)
        _assert_clean(violations, _FLUSH_LOOP_CATEGORY)

    def test_no_bespoke_httpx_clients(self) -> None:
        """no class holds a raw long-lived httpx client field."""
        scan_roots = find_local_src_roots(_REPO_ROOT)
        violations = find_bespoke_httpx_clients(scan_roots, _REPO_ROOT, _SANCTIONED_HTTPX_SITES)
        _assert_clean(violations, _HTTPX_CLIENT_CATEGORY)

    def test_no_bespoke_cache_wrappers(self) -> None:
        """no class wraps SQLiteBackend + cache verbs without subclassing BaseCollection."""
        scan_roots = find_local_src_roots(_REPO_ROOT)
        inheritance_roots = discover_src_roots(_REPO_ROOT)
        violations = find_wrapper_classes(
            scan_roots,
            _REPO_ROOT,
            inheritance_roots,
            _BASE_COLLECTION_NAMES,
            _CACHE_METHOD_NAMES,
        )
        _assert_clean(violations, "reuse.bespoke_cache_wrapper")


class TestWalkersFlagPlantedViolations:
    """each walker must fire on a planted violation (self-test of the guard)."""

    def test_flush_loop_walker_flags_planted(self, tmp_path: Path) -> None:
        source = (
            "import asyncio\n\n\n"
            "class BespokeBuffer:\n"
            "    def __init__(self) -> None:\n"
            "        self._pending: list[str] = []\n\n"
            "    def enqueue(self, item: str) -> None:\n"
            "        self._pending.append(item)\n\n"
            "    async def _loop(self) -> None:\n"
            "        while True:\n"
            "            await asyncio.sleep(5)\n"
            "            self._pending.clear()\n"
        )
        planted = tmp_path / "bespoke_buffer.py"
        planted.write_text(source, encoding="utf-8")
        violations = find_bespoke_flush_loops((tmp_path,), tmp_path, frozenset())
        assert [v.symbol for v in violations] == ["BespokeBuffer"]

    def test_httpx_walker_flags_planted(self, tmp_path: Path) -> None:
        source = (
            "import httpx\n\n\n"
            "class BespokeApiClient:\n"
            "    def __init__(self) -> None:\n"
            "        self._client = httpx.AsyncClient(timeout=30)\n"
        )
        planted = tmp_path / "bespoke_client.py"
        planted.write_text(source, encoding="utf-8")
        violations = find_bespoke_httpx_clients((tmp_path,), tmp_path, frozenset())
        assert [v.symbol for v in violations] == ["BespokeApiClient"]

    def test_wrapper_walker_flags_planted(self, tmp_path: Path) -> None:
        source = (
            "from threetears.core.cache.sqlite import SQLiteBackend\n\n\n"
            "class BespokeCounter:\n"
            "    def __init__(self, backend: SQLiteBackend) -> None:\n"
            "        self._backend = backend\n\n"
            "    async def get(self, key: str) -> int:\n"
            "        return 0\n\n"
            "    async def put(self, key: str, value: int) -> None:\n"
            "        return None\n"
        )
        planted = tmp_path / "bespoke_counter.py"
        planted.write_text(source, encoding="utf-8")
        violations = find_wrapper_classes(
            (tmp_path,),
            tmp_path,
            (tmp_path,),
            _BASE_COLLECTION_NAMES,
            _CACHE_METHOD_NAMES,
        )
        assert [v.symbol for v in violations] == ["BespokeCounter"]
